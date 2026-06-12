"""Threshold-policy shoot-out on the real temporal split, processed in time order.

Compares three alerting policies on identical MLX-autoencoder scores:
  static     — p99 of the benign calibration set (the original deployed policy)
  conformal  — split-conformal p-value <= alpha (finite-sample FPR guarantee
               under exchangeability; static threshold equivalent)
  budget     — label-free online alert-rate controller (adapted ACI)

Usage: python scripts/eval_conformal.py [--alpha 0.01] [--gamma 0.005]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.config import get_settings
from sentinel.ids.conformal import AlertBudgetController, conformal_pvalues, empirical_fpr
from sentinel.ids.data import DAY_COLUMN, FlowScaler, load_flows, make_xy
from sentinel.ids.train import TRAIN_DAYS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    from sentinel.ids.anomaly_mlx import reconstruction_errors_mlx, train_autoencoder_mlx

    settings = get_settings()
    flows = load_flows(args.data_dir or settings.ids_data_dir, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    x_train, y_train, _ = make_xy(flows.loc[in_train], attempted="drop")
    x_test, y_test, labels_test = make_xy(flows.loc[~in_train], attempted="drop")

    benign = x_train.loc[y_train == 0]
    holdout = benign.sample(frac=0.1, random_state=args.seed)
    scaler = FlowScaler().fit(benign.drop(index=holdout.index))
    model = train_autoencoder_mlx(scaler.transform(benign.drop(index=holdout.index)))
    calibration = reconstruction_errors_mlx(model, scaler.transform(holdout))

    # Time-order the test stream: drift unfolds chronologically.
    timestamps = pd.to_datetime(
        flows.loc[x_test.index, "Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce"
    )
    order = np.argsort(timestamps.to_numpy())
    scores = reconstruction_errors_mlx(model, scaler.transform(x_test))[order]
    y = y_test.to_numpy()[order]
    labels = labels_test.to_numpy()[order]
    is_benign = y == 0

    def report(name: str, alerts: np.ndarray) -> None:
        recall = float(alerts[~is_benign].mean())
        print(
            f"{name:<10} alert_rate={alerts.mean():.4f}  "
            f"FPR={empirical_fpr(alerts, is_benign):.4f}  recall={recall:.4f}"
        )
        for label in ("DDoS", "Infiltration", "Web Attack - XSS"):
            mask = labels == label
            if mask.any():
                print(f"{'':<12}recall {label}: {float(alerts[mask].mean()):.3f}")

    static_alerts = scores > np.percentile(calibration, 100 * (1 - args.alpha))
    report("static", static_alerts)

    conformal_alerts = conformal_pvalues(calibration, scores) <= args.alpha
    report("conformal", conformal_alerts)

    result = AlertBudgetController(calibration, alpha=args.alpha, gamma=args.gamma).run(scores)
    report("budget", result.alerts)
    print(f"\nbudget alpha_t final={result.alpha_path[-1]:.4f}")


if __name__ == "__main__":
    main()
