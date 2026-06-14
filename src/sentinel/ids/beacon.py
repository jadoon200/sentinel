"""Beacon detector by data-size dispersion — the signature periodicity missed.

The recorded arc (docs/EVAL.md) is three failed beacon attempts, all framed on
**timing periodicity**: inter-arrival CV (profile per-pair, Bot recall 0.056),
and the Schuster periodogram (spectral, AUC 0.73, recall 0.000). The honest
conclusion was that periodicity is a benign-infrastructure trap — NTP and
keepalives are *more* periodic than a jittered attacker beacon.

This detector changes the frame. An ARES C2 channel interleaves payload-less
poll flows ("Bot - Attempted" — 67% of 2017 Bot flows) with data-carrying
tasking flows, so a single (src→dst) channel's forward-payload sizes are wildly
**dispersed**; a benign periodic service (NTP) sends a uniform packet every
time, so its sizes barely move. Scoring each channel by the dispersion
(coefficient of variation) of its forward bytes and mean packet length —
benign-calibrated, thresholded at a benign-channel percentile like everywhere
in SENTINEL — separates the C2 channels the timing detectors could only rank.

On CIC-IDS2017 this lifts Bot channel recall from ~0 to 5/5 at a ~1% benign
false-positive rate (AUC ~0.998). Validated at the channel level on only the
five 2017 C2 channels, so it is a strong foothold rather than a closed gap; the
*mechanism* (≈50% empty polls + ≈50% data flows) is confirmed on CSE-CIC-IDS2018
Bot's 286k flows, whose public CSVs drop IPs so the channel statistic cannot be
recomputed there. Dispersion uses behavioral size statistics only — never an
IP, port, or timestamp — so the no-topology-leak rule still holds.

Usage:
    python -m sentinel.ids.beacon [--min-events 16]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SRC_COLUMN = "Src IP"
DST_IP_COLUMN = "Dst IP"
TS_COLUMN = "Timestamp"
TS_FORMAT = "%d/%m/%Y %I:%M:%S %p"
# Forward-payload byte total (header drifts between dataset variants).
FWD_BYTES_COLUMNS = ("Total Length of Fwd Packet", "Total Length of Fwd Packets")
PKT_LEN_COLUMN = "Packet Length Mean"

# The dispersion statistics scored per channel (named for explainability).
STAT_NAMES = ["fwd_bytes_cv", "pkt_len_cv"]


def _cv(values: NDArray[np.float64]) -> float:
    """Coefficient of variation (std / mean); 0 when the mean is non-positive."""
    mean = float(np.nanmean(values))
    return float(np.nanstd(values) / mean) if mean > 0 else 0.0


def _fwd_bytes_column(flows: pd.DataFrame) -> str:
    for name in FWD_BYTES_COLUMNS:
        if name in flows.columns:
            return name
    raise KeyError(f"no forward-bytes column found (tried {FWD_BYTES_COLUMNS})")


def channel_dispersion(flows: pd.DataFrame, min_events: int = 16) -> pd.DataFrame:
    """Per (src→dst) channel size-dispersion stats for channels with ≥ min_events flows.

    Returns src, dst, n_events, the dispersion stats in STAT_NAMES, and the
    positional index of the channel's last flow (for labeling).
    """
    frame = pd.DataFrame(
        {
            "src": flows[SRC_COLUMN].to_numpy(),
            "dst": flows[DST_IP_COLUMN].to_numpy(),
            "ts": pd.to_datetime(flows[TS_COLUMN], format=TS_FORMAT, errors="coerce"),
            "bytes": pd.to_numeric(flows[_fwd_bytes_column(flows)], errors="coerce"),
            "pkt_len": pd.to_numeric(flows[PKT_LEN_COLUMN], errors="coerce"),
            "pos": np.arange(len(flows)),
        }
    ).dropna(subset=["ts"])

    rows = []
    for (src, dst), group in frame.groupby(["src", "dst"], sort=False):
        if len(group) < min_events:
            continue
        ordered = group.sort_values("ts", kind="stable")
        rows.append(
            {
                "src": src,
                "dst": dst,
                "n_events": len(group),
                "fwd_bytes_cv": _cv(ordered["bytes"].to_numpy(dtype=float)),
                "pkt_len_cv": _cv(ordered["pkt_len"].to_numpy(dtype=float)),
                "last_pos": int(ordered["pos"].iloc[-1]),
            }
        )
    if not rows:
        raise ValueError("no (src, dst) channel reaches min_events")
    return pd.DataFrame(rows)


class BeaconScorer:
    """Max robust-z over the dispersion statistics, calibrated on benign channels.

    Same transparent rule as the host-profile detector: a channel alerts when any
    of its size-dispersion statistics is extreme relative to the benign median.
    """

    def __init__(self) -> None:
        self.median: NDArray[np.float64] | None = None
        self.iqr: NDArray[np.float64] | None = None

    def fit(self, benign_stats: NDArray[np.float64]) -> "BeaconScorer":
        self.median = np.median(benign_stats, axis=0)
        q75, q25 = np.percentile(benign_stats, [75, 25], axis=0)
        self.iqr = np.where((q75 - q25) > 0, q75 - q25, 1.0)
        return self

    def score(self, stats: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self.median is not None and self.iqr is not None
        z = (stats - self.median) / self.iqr
        return np.asarray(z.max(axis=1))  # one-sided: only excess dispersion alerts


def _stats_matrix(channels: pd.DataFrame) -> NDArray[np.float64]:
    return channels[STAT_NAMES].to_numpy(dtype=np.float64)


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--min-events", type=int, default=16)
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
    mlflow.set_experiment("ids-beacon")

    flows = load_flows(args.data_dir or settings.ids_data_dir, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    _, y_train, _ = make_xy(flows.loc[in_train], attempted="drop")
    # Payload-less C2 polls carry the "- Attempted" label but ARE half the beacon
    # signature (the dispersion needs both the empty polls and the data flows),
    # so the test side keeps attempted flows — same policy as the spectral detector.
    x_test, _y_test, labels_test = make_xy(flows.loc[~in_train], attempted="malicious")

    benign_flows = flows.loc[y_train.index[y_train == 0]].reset_index(drop=True)
    benign = channel_dispersion(benign_flows, args.min_events)
    scorer = BeaconScorer().fit(_stats_matrix(benign))
    threshold = float(np.percentile(scorer.score(_stats_matrix(benign)), args.threshold_percentile))

    test_flows = flows.loc[x_test.index].reset_index(drop=True)
    channels = channel_dispersion(test_flows, args.min_events)
    scores = scorer.score(_stats_matrix(channels))
    alerts = scores > threshold

    # Label a channel by its dominant attack label (the ARES victims keep
    # beaconing past the labeled window, so channels can END on benign flows —
    # the same ground-truth gap the spectral detector documents).
    per_flow = pd.DataFrame(
        {
            "src": test_flows[SRC_COLUMN].to_numpy(),
            "dst": test_flows[DST_IP_COLUMN].to_numpy(),
            "label": labels_test.to_numpy(),
        }
    )
    attack_mode = (
        per_flow[per_flow["label"].str.upper() != "BENIGN"]
        .groupby(["src", "dst"])["label"]
        .agg(lambda x: x.mode().iloc[0])
    )
    keys = list(zip(channels["src"], channels["dst"], strict=True))
    channel_labels = np.array([attack_mode.get(k, "BENIGN") for k in keys])
    channel_y = (np.char.upper(channel_labels.astype(str)) != "BENIGN").astype(int)
    print(f"benign channels {len(benign)}, test channels {len(channels)}")

    metrics = {
        "roc_auc": float(roc_auc_score(channel_y, scores)) if channel_y.any() else 0.0,
        "false_positive_rate": float(alerts[channel_y == 0].mean()),
        "recall_overall": float(alerts[channel_y == 1].mean()) if channel_y.any() else 0.0,
    }
    for label in sorted(np.unique(channel_labels[channel_y == 1])):
        mask = channel_labels == label
        metrics[f"recall__{str(label).replace(' ', '_')}"] = float(alerts[mask].mean())

    with mlflow.start_run():
        mlflow.log_params(
            {
                "min_events": args.min_events,
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
