"""Host-profile detector: per-window fan-out statistics (no neural net needed).

The sequence model's recorded negative (docs/EVAL.md): scans and beacons are
*more predictable* than benign traffic, so prediction error can't see them.
What does distinguish them is fan-out — one host touching hundreds of distinct
destination ports/hosts in a short window. This detector computes per-window
cardinality/rate statistics over each source host's time-ordered flows,
robust-scales them on benign Mon-Wed windows, and alerts when any statistic
is extreme. Destination port/IP are used only inside window *counts* — never
as raw feature values — so testbed topology still cannot leak.

Usage:
    python -m sentinel.ids.profile [--window 16] [--stride 8]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SRC_COLUMN = "Src IP"
DST_IP_COLUMN = "Dst IP"
DST_PORT_COLUMN = "Dst Port"
TS_COLUMN = "Timestamp"
TS_FORMAT = "%d/%m/%Y %I:%M:%S %p"

STAT_NAMES = ["unique_dst_ports", "unique_dst_ips", "log_flows_per_sec", "log_mean_fwd_pkts"]


def build_window_stats(
    flows: pd.DataFrame,
    window: int = 16,
    stride: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Per-host sliding-window fan-out stats; returns (N, n_stats) and last positions."""
    # Header drifts between dataset variants: singular vs plural packet column.
    pkts_column = "Total Fwd Packet" if "Total Fwd Packet" in flows.columns else "Total Fwd Packets"
    order = pd.DataFrame(
        {
            "host": flows[SRC_COLUMN].to_numpy(),
            "ts": pd.to_datetime(flows[TS_COLUMN], format=TS_FORMAT, errors="coerce"),
            "dst_port": flows[DST_PORT_COLUMN].to_numpy(),
            "dst_ip": flows[DST_IP_COLUMN].to_numpy(),
            "fwd_pkts": pd.to_numeric(flows[pkts_column], errors="coerce").to_numpy(),
            "pos": np.arange(len(flows)),
        }
    )
    stats = []
    last_positions = []
    for _, group in order.groupby("host", sort=False):
        g = group.sort_values("ts", kind="stable")
        n = len(g)
        for start in range(0, n - window + 1, stride):
            chunk = g.iloc[start : start + window]
            span = (chunk["ts"].iloc[-1] - chunk["ts"].iloc[0]).total_seconds()
            rate = window / max(span, 1.0)
            stats.append(
                [
                    float(chunk["dst_port"].nunique()),
                    float(chunk["dst_ip"].nunique()),
                    float(np.log1p(rate)),
                    float(np.log1p(np.nanmean(chunk["fwd_pkts"].to_numpy(dtype=float)))),
                ]
            )
            last_positions.append(int(chunk["pos"].iloc[-1]))
    if not stats:
        raise ValueError("no host stream is long enough for the window size")
    return np.asarray(stats, dtype=np.float64), np.asarray(last_positions, dtype=np.int64)


class ProfileScorer:
    """Max robust-z over the window statistics, calibrated on benign windows."""

    def __init__(self) -> None:
        self.median: NDArray[np.float64] | None = None
        self.iqr: NDArray[np.float64] | None = None

    def fit(self, benign_stats: NDArray[np.float64]) -> "ProfileScorer":
        self.median = np.median(benign_stats, axis=0)
        q75, q25 = np.percentile(benign_stats, [75, 25], axis=0)
        self.iqr = np.where((q75 - q25) > 0, q75 - q25, 1.0)
        return self

    def score(self, stats: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.median is not None and self.iqr is not None
        z = (stats - self.median) / self.iqr
        return np.asarray(z.max(axis=1))  # one-sided: only excess fan-out/rate alerts


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    import mlflow
    from sklearn.metrics import roc_auc_score

    from sentinel.config import get_settings
    from sentinel.ids.data import DAY_COLUMN, load_flows, make_xy
    from sentinel.ids.train import TRAIN_DAYS

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ids-profile")

    flows = load_flows(args.data_dir or settings.ids_data_dir, sample=args.sample, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    _, y_train, _ = make_xy(flows.loc[in_train], attempted="drop")
    x_test, y_test, labels_test = make_xy(flows.loc[~in_train], attempted="drop")

    benign_flows = flows.loc[y_train.index[y_train == 0]].reset_index(drop=True)
    benign_stats, _ = build_window_stats(benign_flows, args.window, args.stride)
    rng = np.random.default_rng(args.seed)
    holdout_mask = rng.random(len(benign_stats)) < 0.1
    scorer = ProfileScorer().fit(benign_stats[~holdout_mask])
    threshold = float(
        np.percentile(scorer.score(benign_stats[holdout_mask]), args.threshold_percentile)
    )

    test_flows = flows.loc[x_test.index].reset_index(drop=True)
    test_stats, last_pos = build_window_stats(test_flows, args.window, args.stride)
    scores = scorer.score(test_stats)
    alerts = scores > threshold

    window_y = y_test.to_numpy()[last_pos]
    window_labels = labels_test.to_numpy()[last_pos]
    print(f"benign windows {benign_stats.shape}, test windows {test_stats.shape}")

    metrics = {
        "roc_auc": float(roc_auc_score(window_y, scores)),
        "false_positive_rate": float(alerts[window_y == 0].mean()),
        "recall_overall": float(alerts[window_y == 1].mean()),
    }
    for label in sorted(np.unique(window_labels[window_y == 1])):
        mask = window_labels == label
        metrics[f"recall__{str(label).replace(' ', '_')}"] = float(alerts[mask].mean())

    with mlflow.start_run():
        mlflow.log_params(
            {
                "window": args.window,
                "stride": args.stride,
                "threshold_percentile": args.threshold_percentile,
                "threshold": threshold,
                "stats": ",".join(STAT_NAMES),
            }
        )
        mlflow.log_metrics(metrics)

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
