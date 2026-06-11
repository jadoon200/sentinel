"""Sequence-level anomaly detection over per-host flow streams (MLX).

Per-flow detectors miss attacks whose individual flows look benign: port
scans and bot beacons emit thousands of small, normal-looking flows whose
*pattern over time* is the anomaly (measured: the per-flow autoencoder gets
0.007 recall on PortScan, 0.06 on Bot — docs/EVAL.md). This model orders each
source host's flows in time, slides fixed windows over them, and trains an
input-gated recurrent cell to predict the next flow's features from the
preceding ones — on benign Mon-Wed traffic only. A window's anomaly score is
its mean next-step prediction error; thresholding at a benign-holdout
percentile fixes the false-positive rate.

The gated-recurrence design is adapted from selective-SSM work (input-dependent
gates deciding what history to keep); at window length 16 a stepwise loop is
optimal — parallel-scan kernels only pay at hundreds of steps.

Source IP is used ONLY to group flows into per-host streams — never as a
feature — so testbed topology cannot leak into the score.

Usage:
    python -m sentinel.ids.sequence [--window 16] [--stride 8] [--epochs 3]
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as mlx_optim
import numpy as np
import pandas as pd
from numpy.typing import NDArray

SRC_COLUMN = "Src IP"
TS_COLUMN = "Timestamp"
TS_FORMAT = "%d/%m/%Y %I:%M:%S %p"


def build_windows(
    flows: pd.DataFrame,
    features: NDArray[np.float32],
    window: int = 16,
    stride: int = 8,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Slide windows over each host's time-ordered flows.

    `features` rows must align positionally with `flows` rows. Returns the
    window tensor (N, window, D) and the positional index of each window's
    last flow (for labeling windows by their most recent flow).
    """
    order = pd.DataFrame(
        {
            "host": flows[SRC_COLUMN].to_numpy(),
            "ts": pd.to_datetime(flows[TS_COLUMN], format=TS_FORMAT, errors="coerce"),
            "pos": np.arange(len(flows)),
        }
    )
    windows = []
    last_positions = []
    for _, group in order.groupby("host", sort=False):
        ordered = group.sort_values("ts", kind="stable")
        positions = ordered["pos"].to_numpy()
        # Inter-arrival time within the host's stream: the canonical scan/beacon
        # signature (microsecond gaps, fixed periodicity). log-compressed to the
        # scale of the robust-scaled flow features. Leakage-free: relative, not
        # absolute, time.
        seconds = ordered["ts"].diff().dt.total_seconds().fillna(0.0).to_numpy()
        delta_t = (np.log1p(np.clip(seconds, 0.0, None)) - 2.0).astype(np.float32)
        for start in range(0, len(positions) - window + 1, stride):
            idx = positions[start : start + window]
            stacked = np.concatenate([features[idx], delta_t[start : start + window, None]], axis=1)
            windows.append(stacked)
            last_positions.append(idx[-1])
    if not windows:
        raise ValueError("no host stream is long enough for the window size")
    return np.stack(windows), np.asarray(last_positions, dtype=np.int64)


class GatedFlowRNN(mnn.Module):
    """Input-gated recurrence: the gate decides, per step, what history to keep."""

    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.in_proj = mnn.Linear(n_features, hidden)
        self.gate = mnn.Linear(2 * hidden, hidden)
        self.cand = mnn.Linear(2 * hidden, hidden)
        self.head = mnn.Linear(hidden, n_features)
        self.hidden = hidden

    def __call__(self, x: mx.array) -> mx.array:
        """(B, W, D) -> next-step predictions for steps 1..W-1: (B, W-1, D)."""
        h = mx.zeros((x.shape[0], self.hidden))
        predictions = []
        for t in range(x.shape[1] - 1):
            u = self.in_proj(x[:, t, :])
            hu = mx.concatenate([h, u], axis=-1)
            g = mx.sigmoid(self.gate(hu))
            h = g * h + (1 - g) * mx.tanh(self.cand(hu))
            predictions.append(self.head(h))
        return mx.stack(predictions, axis=1)


def _next_step_mse(model: GatedFlowRNN, batch: mx.array) -> mx.array:
    return mnn.losses.mse_loss(model(batch), batch[:, 1:, :], reduction="mean")


def train_sequence_model(
    windows: NDArray[np.float32],
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 13,
    hidden: int = 64,
) -> GatedFlowRNN:
    mx.random.seed(seed)
    model = GatedFlowRNN(windows.shape[2], hidden=hidden)
    optimizer = mlx_optim.Adam(learning_rate=lr)
    loss_and_grad = mnn.value_and_grad(model, _next_step_mse)
    data = mx.array(windows)
    rng = np.random.default_rng(seed)
    n = len(windows)

    for epoch in range(epochs):
        permutation = mx.array(rng.permutation(n))
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch = data[permutation[start : start + batch_size]]
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss) * batch.shape[0]
        print(f"epoch {epoch + 1}/{epochs} next-step mse={epoch_loss / n:.5f}")
    return model


def window_scores(
    model: GatedFlowRNN, windows: NDArray[np.float32], batch_size: int = 2048
) -> NDArray[np.float64]:
    scores = []
    for start in range(0, len(windows), batch_size):
        batch = mx.array(windows[start : start + batch_size])
        error = ((model(batch) - batch[:, 1:, :]) ** 2).mean(axis=(1, 2))
        mx.eval(error)
        scores.append(np.asarray(error))
    return np.concatenate(scores).astype(np.float64)


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    import mlflow
    from sklearn.metrics import roc_auc_score

    from sentinel.config import get_settings
    from sentinel.ids.data import DAY_COLUMN, FlowScaler, load_flows, make_xy
    from sentinel.ids.train import TRAIN_DAYS

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ids-sequence")

    flows = load_flows(args.data_dir or settings.ids_data_dir, sample=args.sample, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    x_train, y_train, _ = make_xy(flows.loc[in_train], attempted="drop")
    x_test, y_test, labels_test = make_xy(flows.loc[~in_train], attempted="drop")

    benign_index = y_train.index[y_train == 0]
    scaler = FlowScaler().fit(x_train.loc[benign_index])

    benign_flows = flows.loc[benign_index].reset_index(drop=True)
    benign_feats = scaler.transform(x_train.loc[benign_index])
    train_windows, _ = build_windows(benign_flows, benign_feats, args.window, args.stride)
    rng = np.random.default_rng(args.seed)
    holdout_mask = rng.random(len(train_windows)) < 0.1
    fit_windows = train_windows[~holdout_mask]
    print(f"train windows {fit_windows.shape}, holdout {int(holdout_mask.sum())}")

    model = train_sequence_model(fit_windows, epochs=args.epochs, seed=args.seed)
    holdout_scores = window_scores(model, train_windows[holdout_mask])
    threshold = float(np.percentile(holdout_scores, args.threshold_percentile))
    # Hyper-regular attack traffic (scans, beacons) is MORE predictable than
    # benign traffic, so suspiciously low error is alerted too (two-sided).
    low_threshold = float(np.percentile(holdout_scores, args.low_percentile))

    test_flows = flows.loc[x_test.index].reset_index(drop=True)
    test_feats = scaler.transform(x_test)
    test_windows, last_pos = build_windows(test_flows, test_feats, args.window, args.stride)
    scores = window_scores(model, test_windows)
    alerts = (scores > threshold) | (scores < low_threshold)

    window_y = y_test.to_numpy()[last_pos]
    window_labels = labels_test.to_numpy()[last_pos]
    print(f"test windows {test_windows.shape}")

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
                "epochs": args.epochs,
                "threshold_percentile": args.threshold_percentile,
                "threshold": threshold,
                "low_threshold": low_threshold,
                "n_train_windows": len(fit_windows),
                "n_test_windows": len(test_windows),
            }
        )
        mlflow.log_metrics(metrics)

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
