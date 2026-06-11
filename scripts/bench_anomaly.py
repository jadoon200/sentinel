"""Head-to-head: torch-MPS vs MLX flow-autoencoder backends on identical data.

Same architecture, protocol, and seed policy; measures wall-clock for training
and scoring plus metric parity (ROC-AUC / recall@p99 on the temporal split).
The MLX backend becomes the default only if this shows equal-or-better metrics
at lower wall-clock — recorded in docs/EVAL.md.

Usage (inside the sentinel conda env, dataset under data/cicids2017/):
    python scripts/bench_anomaly.py [--epochs 5] [--sample N]
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.config import get_settings
from sentinel.ids.data import DAY_COLUMN, load_flows, make_xy
from sentinel.ids.train import TRAIN_DAYS


def evaluate(
    errors_holdout: np.ndarray, errors_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    threshold = float(np.percentile(errors_holdout, 99.0))
    alerts = errors_test > threshold
    return {
        "roc_auc": float(roc_auc_score(y_test, errors_test)),
        "recall@p99": float(alerts[y_test == 1].mean()),
        "fpr@p99": float(alerts[y_test == 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    settings = get_settings()
    flows = load_flows(args.data_dir or settings.ids_data_dir, sample=args.sample, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    x_train, y_train, _ = make_xy(flows.loc[in_train], attempted="drop")
    x_test, y_test, _ = make_xy(flows.loc[~in_train], attempted="drop")

    from sentinel.ids.anomaly import FlowScaler

    benign = x_train.loc[y_train == 0]
    holdout = benign.sample(frac=0.1, random_state=args.seed)
    fit_set = benign.drop(index=holdout.index)
    scaler = FlowScaler().fit(fit_set)
    train_arr = scaler.transform(fit_set)
    holdout_arr = scaler.transform(holdout)
    test_arr = scaler.transform(x_test)
    y = y_test.to_numpy()
    print(f"train {train_arr.shape}, test {test_arr.shape}\n")

    results: dict[str, dict[str, Any]] = {}

    # --- torch (MPS) ---
    from sentinel.ids.anomaly import reconstruction_errors, train_autoencoder

    t0 = time.perf_counter()
    torch_model = train_autoencoder(train_arr, epochs=args.epochs, seed=args.seed)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hold_err = reconstruction_errors(torch_model, holdout_arr)
    test_err = reconstruction_errors(torch_model, test_arr)
    score_s = time.perf_counter() - t0
    results["torch-mps"] = {
        "train_s": train_s,
        "score_s": score_s,
        **evaluate(hold_err, test_err, y),
    }

    # --- mlx (Metal) ---
    try:
        from sentinel.ids.anomaly_mlx import reconstruction_errors_mlx, train_autoencoder_mlx
    except ImportError:
        print("mlx not installed — torch results only")
        _report(results)
        return

    t0 = time.perf_counter()
    mlx_model = train_autoencoder_mlx(train_arr, epochs=args.epochs, seed=args.seed)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hold_err = reconstruction_errors_mlx(mlx_model, holdout_arr)
    test_err = reconstruction_errors_mlx(mlx_model, test_arr)
    score_s = time.perf_counter() - t0
    results["mlx-metal"] = {
        "train_s": train_s,
        "score_s": score_s,
        **evaluate(hold_err, test_err, y),
    }

    _report(results)


def _report(results: dict[str, dict[str, Any]]) -> None:
    keys = ["train_s", "score_s", "roc_auc", "recall@p99", "fpr@p99"]
    print(f"\n{'backend':<12}" + "".join(f"{k:>12}" for k in keys))
    for backend, row in results.items():
        print(f"{backend:<12}" + "".join(f"{row[k]:>12.3f}" for k in keys))
    if len(results) == 2:
        torch_row, mlx_row = results["torch-mps"], results["mlx-metal"]
        print(
            f"\nspeedup: train {torch_row['train_s'] / mlx_row['train_s']:.2f}x, "
            f"score {torch_row['score_s'] / mlx_row['score_s']:.2f}x"
        )


if __name__ == "__main__":
    main()
