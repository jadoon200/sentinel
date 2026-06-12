"""End-to-end smoke tests: every training CLI runs on a tiny synthetic dataset.

These execute the real main() entry points (argument parsing, data loading,
training, MLflow logging) in seconds, so pipeline wiring regressions surface
in CI rather than on the first real run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlx.core")


def _day_csv(path: Path, day: str, n: int, labels: list[str], seed: int) -> None:
    rng = np.random.default_rng(seed)
    hosts = [f"10.0.{i % 4}.{i % 7 + 1}" for i in range(n)]
    frame = pd.DataFrame(
        {
            "Flow ID": [f"{day}-{i}" for i in range(n)],
            "Src IP": hosts,
            "Src Port": rng.integers(1024, 65000, n),
            "Dst IP": ["192.168.10.50"] * n,
            "Dst Port": rng.integers(1, 1024, n),
            "Protocol": rng.choice([6, 17], n),
            "Timestamp": [
                f"0{3 if day != 'Monday' else 1}/07/2017 "
                f"{(9 + i // 3600) % 12 + 1:02d}:{(i // 60) % 60:02d}:{i % 60:02d} PM"
                for i in range(n)
            ],
            "Flow Duration": rng.exponential(1e5, n),
            "Total Fwd Packet": rng.integers(1, 40, n),
            "Total Bwd packets": rng.integers(0, 40, n),
            "Label": [labels[i % len(labels)] for i in range(n)],
        }
    )
    frame.to_csv(path / f"{day}-WorkingHours.csv", index=False)


@pytest.fixture
def tiny_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "cicids"
    data.mkdir()
    _day_csv(data, "Monday", 240, ["BENIGN"], seed=1)
    _day_csv(data, "Tuesday", 240, ["BENIGN", "BENIGN", "DoS Hulk"], seed=2)
    _day_csv(data, "Thursday", 240, ["BENIGN", "PortScan", "Web Attack - XSS"], seed=3)
    monkeypatch.setenv("SENTINEL_MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    return data


def test_train_main_calibrated_temporal(tiny_dataset: Path) -> None:
    from sentinel.ids.train import main

    metrics = main(
        ["--data-dir", str(tiny_dataset), "--split", "temporal", "--calibrate-fpr", "0.05"]
    )

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "recall__PortScan" in metrics


def test_anomaly_main(tiny_dataset: Path) -> None:
    from sentinel.ids.anomaly import main

    metrics = main(["--data-dir", str(tiny_dataset), "--epochs", "1"])

    assert "recall_overall" in metrics and "false_positive_rate" in metrics


def test_sequence_main(tiny_dataset: Path) -> None:
    from sentinel.ids.sequence import main

    metrics = main(
        ["--data-dir", str(tiny_dataset), "--epochs", "1", "--window", "4", "--stride", "2"]
    )

    assert "roc_auc" in metrics


def test_profile_main(tiny_dataset: Path) -> None:
    from sentinel.ids.profile import main

    metrics = main(["--data-dir", str(tiny_dataset), "--window", "4", "--stride", "2"])

    assert "recall_overall" in metrics
