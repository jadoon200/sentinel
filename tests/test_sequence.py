import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlx.core")

from sentinel.ids.sequence import build_windows, train_sequence_model, window_scores


def _flows_frame() -> tuple[pd.DataFrame, np.ndarray]:
    # Two hosts with interleaved timestamps; features encode (host, step) so
    # window membership and ordering are checkable.
    rows = []
    features = []
    for step in range(6):
        for host_id, host in enumerate(["10.0.0.1", "10.0.0.2"]):
            rows.append(
                {
                    "Src IP": host,
                    "Timestamp": f"03/07/2017 01:0{step}:00 PM",
                }
            )
            features.append([float(host_id), float(step)])
    return pd.DataFrame(rows), np.asarray(features, dtype=np.float32)


def test_build_windows_groups_per_host_in_time_order() -> None:
    flows, features = _flows_frame()

    windows, last_pos = build_windows(flows, features, window=4, stride=2)

    assert windows.shape == (4, 4, 3)  # two hosts x two windows each, +delta-t col
    for w in windows:
        assert len(np.unique(w[:, 0])) == 1  # never mixes hosts
        assert list(w[:, 1]) == sorted(w[:, 1])  # time-ordered steps
    assert windows[0][0, 2] == -2.0  # host's stream opener: delta-t = log1p(0) - 2
    assert abs(windows[0][1, 2] - (np.log1p(60.0) - 2.0)) < 1e-5  # 60s gap encoded
    # last flow of host 0's first window is its step-3 flow
    assert features[last_pos[0]].tolist() == [0.0, 3.0]


def test_sequence_model_flags_pattern_breaks() -> None:
    rng = np.random.default_rng(13)
    t = np.arange(12, dtype=np.float32)
    # benign: smooth per-host ramps; anomalous: white noise (same marginal scale)
    benign = np.stack(
        [
            np.stack([t / 12 + rng.normal(0, 0.05, 12).astype(np.float32)] * 4, axis=1)
            for _ in range(300)
        ]
    )
    noise = rng.normal(0.5, 0.3, (60, 12, 4)).astype(np.float32)

    model = train_sequence_model(benign, epochs=15, batch_size=64)
    benign_scores = window_scores(model, benign)
    noise_scores = window_scores(model, noise)

    threshold = np.percentile(benign_scores, 99)
    assert (noise_scores > threshold).mean() > 0.9
