"""MLX backend for the flow autoencoder (Apple-silicon native, no OpenMP).

Same architecture and training protocol as the torch backend in
sentinel.ids.anomaly. It exists for two measured reasons:

1. MLX links no libomp, so it coexists with LightGBM in one process — the
   torch backend deadlocks there in either import order (see replay.py).
2. Unified memory + lazy evaluation cut per-step overhead for small MLPs
   versus torch-MPS.

It becomes the default only where scripts/bench_anomaly.py shows it
equal-or-faster at metric parity. Optional dependency (macOS only): install
via `make install` on darwin or `pip install -r requirements-mlx.txt`.
"""

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as mlx_optim
import numpy as np
from numpy.typing import NDArray


class FlowAutoencoderMLX(mnn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.encoder = mnn.Sequential(
            mnn.Linear(n_features, 64), mnn.ReLU(), mnn.Linear(64, 16), mnn.ReLU()
        )
        self.decoder = mnn.Sequential(mnn.Linear(16, 64), mnn.ReLU(), mnn.Linear(64, n_features))

    def __call__(self, x: mx.array) -> mx.array:
        return self.decoder(self.encoder(x))


def _mse(model: FlowAutoencoderMLX, batch: mx.array) -> mx.array:
    return mnn.losses.mse_loss(model(batch), batch, reduction="mean")


def train_autoencoder_mlx(
    benign: NDArray[np.float32],
    epochs: int = 5,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 13,
) -> FlowAutoencoderMLX:
    mx.random.seed(seed)
    model = FlowAutoencoderMLX(benign.shape[1])
    optimizer = mlx_optim.Adam(learning_rate=lr)
    loss_and_grad = mnn.value_and_grad(model, _mse)
    data = mx.array(benign)
    rng = np.random.default_rng(seed)
    n = len(benign)

    for epoch in range(epochs):
        permutation = mx.array(rng.permutation(n))
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch = data[permutation[start : start + batch_size]]
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss) * batch.shape[0]
        print(f"epoch {epoch + 1}/{epochs} mse={epoch_loss / n:.5f}")
    return model


def reconstruction_errors_mlx(
    model: FlowAutoencoderMLX, values: NDArray[np.float32], batch_size: int = 8192
) -> NDArray[np.float64]:
    errors = []
    for start in range(0, len(values), batch_size):
        batch = mx.array(values[start : start + batch_size])
        per_row = ((model(batch) - batch) ** 2).mean(axis=1)
        mx.eval(per_row)
        errors.append(np.asarray(per_row))
    return np.concatenate(errors).astype(np.float64)
