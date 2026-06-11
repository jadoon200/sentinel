"""Replay corrected CIC-IDS2017 flows through both IDS models into alerts.

Trains on Mon-Wed, replays Thu-Fri (the temporal split), and persists the
highest-confidence detections as Alert rows tagged with ATT&CK techniques:

- a multiclass LightGBM predicts the attack *family*, which maps to techniques
  via the curated CIC -> ATT&CK map (sentinel.ids.attack_map);
- the benign-only autoencoder contributes anomaly alerts for flows the
  supervised model scores as benign (no technique tag — that's the point).

Usage (inside the sentinel conda env, Postgres up via `make up`):
    python -m sentinel.ids.replay [--sample 500000] [--max-alerts 1000]
"""

import argparse
import os
from pathlib import Path
from typing import Any

# torch and lightgbm both vendor libomp on macOS; running both in one process
# deadlocks LightGBM's OpenMP pool without this (same workaround as tests).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.config import get_settings
from sentinel.ids.attack_map import techniques_for_label
from sentinel.ids.data import DAY_COLUMN, load_flows, make_xy
from sentinel.ids.train import TRAIN_DAYS

MULTICLASS_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "verbose": -1,
    "seed": 13,
}


def train_multiclass(
    features: pd.DataFrame, labels: "pd.Series[str]", num_boost_round: int = 200
) -> tuple[lgb.Booster, list[str]]:
    """Train a family classifier; returns the booster and its class names."""
    classes = sorted(labels.unique())
    encoded = labels.map({name: i for i, name in enumerate(classes)})
    params = {**MULTICLASS_PARAMS, "num_class": len(classes)}
    booster = lgb.train(params, lgb.Dataset(features, label=encoded), num_boost_round)
    return booster, classes


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-alerts", type=int, default=1000, help="per model")
    parser.add_argument("--supervised-threshold", type=float, default=0.5)
    parser.add_argument("--anomaly-percentile", type=float, default=99.0)
    parser.add_argument("--anomaly-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    from sqlalchemy import delete

    from sentinel.db.base import session_scope
    from sentinel.db.models import Alert

    settings = get_settings()
    flows = load_flows(args.data_dir or settings.ids_data_dir, sample=args.sample, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    x_train, y_train, labels_train = make_xy(flows.loc[in_train], attempted="drop")
    x_test, _, labels_test = make_xy(flows.loc[~in_train], attempted="drop")
    days_test = flows.loc[~in_train].loc[labels_test.index, DAY_COLUMN]

    booster, classes = train_multiclass(x_train, labels_train)
    probabilities = np.asarray(booster.predict(x_test))
    predicted_idx = probabilities.argmax(axis=1)
    predicted = np.asarray(classes, dtype=object)[predicted_idx]
    confidence = probabilities.max(axis=1)
    is_attack_prediction = (predicted != "BENIGN") & (confidence >= args.supervised_threshold)

    # torch loads only after all LightGBM work is done (macOS libomp clash).
    from sentinel.ids.anomaly import FlowScaler, reconstruction_errors, train_autoencoder

    benign_train = x_train.loc[y_train == 0]
    holdout = benign_train.sample(frac=0.1, random_state=args.seed)
    scaler = FlowScaler().fit(benign_train.drop(index=holdout.index))
    autoencoder = train_autoencoder(
        scaler.transform(benign_train.drop(index=holdout.index)),
        epochs=args.anomaly_epochs,
        seed=args.seed,
    )
    threshold = float(
        np.percentile(
            reconstruction_errors(autoencoder, scaler.transform(holdout)),
            args.anomaly_percentile,
        )
    )
    errors = reconstruction_errors(autoencoder, scaler.transform(x_test))
    # Anomaly alerts only where the supervised model saw nothing — the
    # ensemble's job is covering unseen attack families, not double-alerting.
    is_anomaly = (errors > threshold) & ~is_attack_prediction

    alerts: list[Alert] = []
    supervised_order = np.argsort(-confidence)
    taken = 0
    for i in supervised_order:
        if taken >= args.max_alerts:
            break
        if not is_attack_prediction[i]:
            continue
        alerts.append(
            Alert(
                model="lightgbm-multiclass",
                day=str(days_test.iloc[i]),
                score=float(confidence[i]),
                predicted_label=str(predicted[i]),
                true_label=str(labels_test.iloc[i]),
                techniques=techniques_for_label(str(predicted[i])),
            )
        )
        taken += 1

    anomaly_order = np.argsort(-errors)
    taken = 0
    for i in anomaly_order:
        if taken >= args.max_alerts:
            break
        if not is_anomaly[i]:
            continue
        alerts.append(
            Alert(
                model="autoencoder",
                day=str(days_test.iloc[i]),
                score=float(errors[i]),
                predicted_label=None,
                true_label=str(labels_test.iloc[i]),
                techniques=[],
            )
        )
        taken += 1

    with session_scope() as session:
        # Alerts are a derived artifact of one replay pass — rebuild wholesale.
        session.execute(delete(Alert))
        for alert in alerts:
            session.add(alert)

    counts = {
        "supervised_alerts": sum(a.model == "lightgbm-multiclass" for a in alerts),
        "anomaly_alerts": sum(a.model == "autoencoder" for a in alerts),
    }
    print(counts)
    return counts


if __name__ == "__main__":
    main()
