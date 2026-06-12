"""Train the LightGBM baseline on corrected CIC-IDS2017 flows, tracked in MLflow.

Usage (inside the sentinel conda env, after downloading the dataset):
    python -m sentinel.ids.train [--sample 500000] [--attempted drop]
"""

import argparse
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sentinel.config import get_settings
from sentinel.ids.data import DAY_COLUMN, AttemptedPolicy, load_flows, make_xy

# Temporal split: train on the first three capture days, test on the last two.
# Thursday/Friday attacks (web, infiltration, botnet, portscan, DDoS) are absent
# from training, so this measures detection of unseen attack families.
TRAIN_DAYS = ["Monday", "Tuesday", "Wednesday"]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "learning_rate": 0.1,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 13,
}


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: "pd.Series[int]",
    x_valid: pd.DataFrame,
    y_valid: "pd.Series[int]",
    params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)
    return lgb.train(
        params or DEFAULT_PARAMS,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )


def evaluate(
    model: lgb.Booster,
    x_test: pd.DataFrame,
    y_test: "pd.Series[int]",
    labels: "pd.Series[str]",
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = np.asarray(model.predict(x_test))
    predicted = scores >= threshold

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "f1": float(f1_score(y_test, predicted)),
        "false_positive_rate": float(predicted[y_test == 0].mean()),
    }
    # Per-attack detection rate: which attack families the model actually catches.
    for label in sorted(labels[y_test == 1].unique()):
        mask = (labels == label).to_numpy()
        metrics[f"recall__{label.replace(' ', '_')}"] = float(predicted[mask].mean())
    return metrics


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--attempted", choices=["drop", "benign", "malicious"], default="drop")
    parser.add_argument("--split", choices=["random", "temporal"], default="random")
    parser.add_argument(
        "--calibrate-fpr",
        type=float,
        default=None,
        help="pick the alert threshold from benign validation scores at this FPR",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    import mlflow

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ids-lightgbm-baseline")

    attempted: AttemptedPolicy = args.attempted
    flows = load_flows(args.data_dir or settings.ids_data_dir, sample=args.sample, seed=args.seed)

    if args.split == "temporal":
        in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
        x_train, y_train, _ = make_xy(flows.loc[in_train], attempted=attempted)
        x_test, y_test, labels_test = make_xy(flows.loc[~in_train], attempted=attempted)
    else:
        features, target, labels = make_xy(flows, attempted=attempted)
        x_train, x_test, y_train, y_test, _, labels_test = train_test_split(
            features,
            target,
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=target,
        )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                **DEFAULT_PARAMS,
                "attempted_policy": attempted,
                "split": args.split,
                "sample": args.sample or "full",
                "n_train": len(x_train),
                "n_test": len(x_test),
            }
        )
        model = train_lightgbm(x_train, y_train, x_valid, y_valid)
        threshold = 0.5
        if args.calibrate_fpr is not None:
            benign_scores = np.asarray(model.predict(x_valid[y_valid == 0]))
            threshold = float(np.quantile(benign_scores, 1 - args.calibrate_fpr))
            mlflow.log_param("calibrated_threshold", threshold)
        metrics = evaluate(model, x_test, y_test, labels_test, threshold=threshold)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, artifact_path="model")

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
