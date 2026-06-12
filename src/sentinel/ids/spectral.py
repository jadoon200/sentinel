"""Spectral beacon detector: Schuster periodogram over per-pair event times.

The recorded arc (docs/EVAL.md): variance-based periodicity statistics left
Bot at 0.056 recall — beacon cadence overlaps benign periodic services when
summarized as inter-arrival CV. The right instrument is spectral: for a
(src→dst) channel's event times t_1..t_n, the Schuster/Rayleigh statistic

    S(P) = |Σ_k exp(2πi t_k / P)|² / n

is ~1 in expectation for a Poisson stream at any period P, and grows toward n
when events lock to period P. Scoring max_P S(P) over a log-spaced period
grid turns "fires on a timer" into a calibrated test statistic — thresholded,
as everywhere in SENTINEL, at a benign-pair percentile.

Usage:
    python -m sentinel.ids.spectral [--min-events 16]
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


def schuster_score(
    event_seconds: NDArray[np.float64],
    periods: NDArray[np.float64],
    window: int = 16,
    stride: int = 8,
) -> float:
    """max over periods P of mean over windows of |Σ exp(2πi t/P)|² / n.

    Design notes, each one a measured pitfall:
    - short windows: sleep-based beacons accumulate jitter, so phase
      coherence survives ~16 events, not hundreds;
    - periods capped at a third of the window span: longer periods make any
      signal look phase-locked;
    - **mean across windows at a fixed period** (not max-of-max): a real
      timer is coherent at the *same* period in every window, while noise
      maxima wander — averaging crushes the multiple-testing noise floor;
    - the period grid floor must exceed the timestamp quantum (see
      default_period_grid): second-resolution captures are perfectly
      periodic at P = 1s.
    """
    n = len(event_seconds)
    per_window = []
    for start in range(0, max(n - window + 1, 1), stride):
        chunk = event_seconds[start : start + window]
        if len(chunk) < window:
            break
        span = chunk[-1] - chunk[0]
        valid_mask = periods <= span / 3.0
        if not valid_mask.any():
            continue
        t = (chunk - chunk[0])[:, None]
        phases = 2.0 * np.pi * t / periods[None, :]
        power = np.cos(phases).sum(axis=0) ** 2 + np.sin(phases).sum(axis=0) ** 2
        row = power / len(chunk)
        row[~valid_mask] = np.nan
        per_window.append(row)
    if not per_window:
        return 0.0
    with np.errstate(invalid="ignore"):
        mean_power = np.nanmean(np.stack(per_window), axis=0)
    return float(np.nanmax(mean_power)) if not np.isnan(mean_power).all() else 0.0


def default_period_grid(low: float = 4.0, high: float = 3600.0, n: int = 64) -> NDArray[np.float64]:
    """Floor of 4s: flow timestamps are second-quantized, and every
    integer-second stream is perfectly phase-locked at P = 1s; sub-4s
    "beacons" are floods, which the profile detector already owns."""
    return np.logspace(np.log10(low), np.log10(high), n)


def pair_scores(
    flows: pd.DataFrame,
    min_events: int = 16,
    periods: NDArray[np.float64] | None = None,
) -> pd.DataFrame:
    """Schuster score per (src, dst) channel with at least min_events flows.

    Returns a frame with src, dst, n_events, score, and the positional index
    of the channel's last flow (for labeling).
    """
    if periods is None:
        periods = default_period_grid()
    order = pd.DataFrame(
        {
            "src": flows[SRC_COLUMN].to_numpy(),
            "dst": flows[DST_IP_COLUMN].to_numpy(),
            "ts": pd.to_datetime(flows[TS_COLUMN], format=TS_FORMAT, errors="coerce"),
            "pos": np.arange(len(flows)),
        }
    ).dropna(subset=["ts"])
    rows = []
    for (src, dst), group in order.groupby(["src", "dst"], sort=False):
        if len(group) < min_events:
            continue
        ordered = group.sort_values("ts", kind="stable")
        seconds = (ordered["ts"] - ordered["ts"].iloc[0]).dt.total_seconds().to_numpy()
        rows.append(
            {
                "src": src,
                "dst": dst,
                "n_events": len(group),
                "score": schuster_score(seconds, periods),
                "last_pos": int(ordered["pos"].iloc[-1]),
            }
        )
    if not rows:
        raise ValueError("no (src, dst) channel reaches min_events")
    return pd.DataFrame(rows)


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
    mlflow.set_experiment("ids-spectral")

    flows = load_flows(args.data_dir or settings.ids_data_dir, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    _, y_train, _ = make_xy(flows.loc[in_train], attempted="drop")
    # Beacons that never receive tasking are labeled "- Attempted" in the
    # corrected dataset (67% of Bot flows) — for beacon hunting that payload-
    # less C2 polling IS the signal, so the test side keeps attempted flows.
    x_test, _y_test, labels_test = make_xy(flows.loc[~in_train], attempted="malicious")

    benign_flows = flows.loc[y_train.index[y_train == 0]].reset_index(drop=True)
    benign = pair_scores(benign_flows, args.min_events)
    threshold = float(np.percentile(benign["score"], args.threshold_percentile))

    test_flows = flows.loc[x_test.index].reset_index(drop=True)
    channels = pair_scores(test_flows, args.min_events)
    alerts = channels["score"].to_numpy() > threshold

    # Label a channel by its dominant attack label, not its last flow: the
    # ARES victims keep beaconing past the labeled attack window, so C2
    # channels END on benign-labeled flows (a second ground-truth gap — the
    # first was the Infiltration victim's lateral scan; see docs/EVAL.md).
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
        "roc_auc": float(roc_auc_score(channel_y, channels["score"])) if channel_y.any() else 0.0,
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
            }
        )
        mlflow.log_metrics(metrics)

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
