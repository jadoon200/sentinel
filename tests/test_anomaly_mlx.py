import numpy as np
import pytest

pytest.importorskip("mlx.core")

from sentinel.ids.anomaly_mlx import reconstruction_errors_mlx, train_autoencoder_mlx


def test_mlx_autoencoder_separates_outliers() -> None:
    rng = np.random.default_rng(13)
    benign = rng.normal(0, 1, (1000, 8)).astype(np.float32)
    anomalies = rng.normal(6, 1, (100, 8)).astype(np.float32)

    model = train_autoencoder_mlx(benign, epochs=20, batch_size=256)
    benign_errors = reconstruction_errors_mlx(model, benign)
    anomaly_errors = reconstruction_errors_mlx(model, anomalies)

    threshold = np.percentile(benign_errors, 99)
    assert (anomaly_errors > threshold).mean() > 0.9
