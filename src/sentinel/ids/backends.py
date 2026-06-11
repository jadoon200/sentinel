"""Anomaly-backend selection: MLX when available (macOS), torch otherwise.

The 5-seed benchmark (scripts/bench_anomaly.py, recorded in docs/EVAL.md)
showed recall@p99 parity at 3.7x faster training — and MLX links no libomp,
so it can share a process with LightGBM where the torch backend deadlocks.
Importing this module loads neither framework; selection defers the import.
"""

from collections.abc import Callable
from typing import Any

TrainFn = Callable[..., Any]
ScoreFn = Callable[..., Any]


def select_anomaly_backend(name: str = "auto") -> tuple[str, TrainFn, ScoreFn]:
    if name in ("auto", "mlx"):
        try:
            from sentinel.ids.anomaly_mlx import (
                reconstruction_errors_mlx,
                train_autoencoder_mlx,
            )

            return "mlx", train_autoencoder_mlx, reconstruction_errors_mlx
        except ImportError:
            if name == "mlx":
                raise
    from sentinel.ids.anomaly import reconstruction_errors, train_autoencoder

    return "torch", train_autoencoder, reconstruction_errors
