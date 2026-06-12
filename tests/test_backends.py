import pytest

from sentinel.ids.backends import select_anomaly_backend


def test_auto_prefers_mlx_when_available() -> None:
    pytest.importorskip("mlx.core")
    name, train_fn, _score_fn = select_anomaly_backend("auto")

    assert name == "mlx"
    assert train_fn.__name__ == "train_autoencoder_mlx"


def test_explicit_torch_backend() -> None:
    name, train_fn, _score_fn = select_anomaly_backend("torch")

    assert name == "torch"
    assert train_fn.__name__ == "train_autoencoder"
