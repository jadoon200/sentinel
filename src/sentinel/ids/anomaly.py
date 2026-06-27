"""Autoencoder anomaly detector for network flows (PyTorch, CPU/MPS).

Complement to the supervised baseline: trained on *benign Mon-Wed flows only*
(no attack labels), it scores flows by reconstruction error, so attack families
never seen during training can still alert. The detection threshold is set to
a chosen percentile of benign validation error, which fixes the false-positive
rate by construction.

Usage:
    python -m sentinel.ids.anomaly [--epochs 5] [--threshold-percentile 99]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sentinel.config import get_settings
from sentinel.ids.data import DAY_COLUMN, FlowScaler, load_flows, make_xy
from sentinel.ids.train import TRAIN_DAYS

# All heavy compute runs on MPS; torch's CPU OpenMP pool deadlocks when
# lightgbm's libomp is already loaded in the process (replay), so keep
# torch CPU-side single-threaded — it costs nothing here.
torch.set_num_threads(1)


def _device() -> torch.device:
    # MPS can report available but be unusable (CI VMs, broken Metal): probe
    # with a real allocation before committing to it.
    if torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except RuntimeError:
            pass
    return torch.device("cpu")


class FlowAutoencoder(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded: torch.Tensor = self.decoder(self.encoder(x))
        return decoded


def train_autoencoder(
    benign: NDArray[np.float32],
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 13,
) -> FlowAutoencoder:
    torch.manual_seed(seed)
    device = _device()
    model = FlowAutoencoder(benign.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data = torch.from_numpy(benign)

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(data))
        epoch_loss = 0.0
        for start in range(0, len(data), batch_size):
            batch = data[permutation[start : start + batch_size]].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
        print(f"epoch {epoch + 1}/{epochs} mse={epoch_loss / len(data):.5f}")
    return model


def reconstruction_errors(
    model: FlowAutoencoder, values: NDArray[np.float32], batch_size: int = 8192
) -> NDArray[np.float64]:
    device = _device()
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            per_row = ((model(batch) - batch) ** 2).mean(dim=1)
            errors.append(per_row.cpu().numpy())
    return np.concatenate(errors).astype(np.float64)


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--conformal",
        action="store_true",
        help="gate alerts with the online budget controller (drift-robust) "
        "instead of the fixed benign percentile",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    import mlflow
    from sklearn.metrics import roc_auc_score

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ids-autoencoder")

    flows = load_flows(args.data_dir or settings.ids_data_dir, sample=args.sample, seed=args.seed)
    in_train = flows[DAY_COLUMN].isin(TRAIN_DAYS)
    x_train, y_train, _ = make_xy(flows.loc[in_train], attempted="drop")
    x_test, y_test, labels_test = make_xy(flows.loc[~in_train], attempted="drop")

    benign = x_train.loc[y_train == 0]
    holdout = benign.sample(frac=0.1, random_state=args.seed)
    fit_set = benign.drop(index=holdout.index)

    scaler = FlowScaler().fit(fit_set)
    model = train_autoencoder(scaler.transform(fit_set), epochs=args.epochs, seed=args.seed)

    holdout_errors = reconstruction_errors(model, scaler.transform(holdout))
    threshold = float(np.percentile(holdout_errors, args.threshold_percentile))
    errors = reconstruction_errors(model, scaler.transform(x_test))
    if args.conformal:
        from sentinel.ids.conformal import budget_alerts

        alerts = budget_alerts(holdout_errors, errors, args.threshold_percentile)
    else:
        alerts = errors > threshold

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, errors)),
        "false_positive_rate": float(alerts[(y_test == 0).to_numpy()].mean()),
        "recall_overall": float(alerts[(y_test == 1).to_numpy()].mean()),
    }
    for label in sorted(labels_test[y_test == 1].unique()):
        mask = (labels_test == label).to_numpy()
        metrics[f"recall__{label.replace(' ', '_')}"] = float(alerts[mask].mean())

    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "threshold_percentile": args.threshold_percentile,
                "threshold": threshold,
                "conformal": args.conformal,
                "n_benign_train": len(fit_set),
                "n_test": len(x_test),
                "sample": args.sample or "full",
            }
        )
        mlflow.log_metrics(metrics)

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
