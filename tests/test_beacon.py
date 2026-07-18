"""Beacon-by-dispersion: a poll+tasking channel out-scores a uniform benign one."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.ids.beacon import BeaconScorer, channel_dispersion


def _channel(
    src: str,
    dst: str,
    byte_sizes: list[int],
    start_hour: int,
    byte_col: str = "Total Length of Fwd Packet",
    label: str | list[str] = "BENIGN",
) -> pd.DataFrame:
    n = len(byte_sizes)
    ts = [f"07/07/2017 {start_hour:02d}:{m:02d}:00 PM" for m in range(n)]
    return pd.DataFrame(
        {
            "Src IP": [src] * n,
            "Dst IP": [dst] * n,
            "Timestamp": ts,
            byte_col: byte_sizes,
            "Packet Length Mean": [b / 2.0 for b in byte_sizes],
            "Label": label if isinstance(label, list) else [label] * n,
        }
    )


def _flows() -> pd.DataFrame:
    # Beacon: 8 empty C2 polls interleaved with 8 data-tasking flows -> high size CV.
    beacon = _channel(
        "10.0.0.5", "203.0.113.9", [0, 4000, 0, 5200, 0, 3800, 0, 6100] * 2, start_hour=1
    )
    # Benign periodic service (e.g. NTP): a uniform packet every time -> ~0 size CV.
    benign = _channel("10.0.0.6", "192.168.1.1", [76] * 16, start_hour=2)
    return pd.concat([beacon, benign], ignore_index=True)


def test_dispersion_separates_beacon_from_uniform_channel() -> None:
    channels = channel_dispersion(_flows(), min_events=16)
    by_dst = channels.set_index("dst")

    # The poll+tasking channel is far more size-dispersed than the uniform one.
    assert by_dst.loc["203.0.113.9", "fwd_bytes_cv"] > 1.0
    assert by_dst.loc["192.168.1.1", "fwd_bytes_cv"] < 0.1


def test_benign_calibrated_scorer_flags_only_the_beacon() -> None:
    channels = channel_dispersion(_flows(), min_events=16)
    stats = channels[["fwd_bytes_cv", "pkt_len_cv"]].to_numpy(dtype=float)
    benign_row = channels["dst"] == "192.168.1.1"

    # Calibrate on the benign (uniform) channel; the beacon must clear its threshold.
    scorer = BeaconScorer().fit(stats[benign_row.to_numpy()])
    scores = scorer.score(stats)
    beacon_score = scores[(channels["dst"] == "203.0.113.9").to_numpy()][0]
    benign_score = scores[benign_row.to_numpy()][0]

    assert beacon_score > benign_score
    assert beacon_score > 1.0 and abs(benign_score) < 1e-9


def test_fwd_bytes_column_accepts_plural_header_variant() -> None:
    # The corrected-dataset header drifts between "Packet" and "Packets" —
    # channel_dispersion must fall back to the plural variant transparently.
    flows = _channel(
        "10.0.0.7",
        "192.168.1.7",
        [100] * 16,
        start_hour=3,
        byte_col="Total Length of Fwd Packets",
    )
    channels = channel_dispersion(flows, min_events=16)
    assert channels.loc[0, "n_events"] == 16


def test_missing_fwd_bytes_column_raises_keyerror() -> None:
    flows = _channel("10.0.0.8", "192.168.1.8", [100] * 16, start_hour=4).drop(
        columns=["Total Length of Fwd Packet"]
    )
    with pytest.raises(KeyError, match="no forward-bytes column found"):
        channel_dispersion(flows, min_events=16)


def test_min_events_filters_out_short_channels() -> None:
    short = _channel("10.0.0.1", "192.168.9.1", [100] * 5, start_hour=1)
    long_enough = _channel("10.0.0.2", "192.168.9.2", [100] * 16, start_hour=2)
    flows = pd.concat([short, long_enough], ignore_index=True)

    channels = channel_dispersion(flows, min_events=16)

    # Only the channel meeting min_events survives; the short one is dropped.
    assert len(channels) == 1
    assert channels.iloc[0]["dst"] == "192.168.9.2"


def test_no_channel_reaching_min_events_raises() -> None:
    flows = _channel("10.0.0.1", "192.168.9.1", [100] * 5, start_hour=1)
    with pytest.raises(ValueError, match=r"no \(src, dst\) channel reaches min_events"):
        channel_dispersion(flows, min_events=16)


def test_empty_frame_raises_value_error() -> None:
    empty = pd.DataFrame(
        {
            "Src IP": pd.Series([], dtype=str),
            "Dst IP": pd.Series([], dtype=str),
            "Timestamp": pd.Series([], dtype=str),
            "Total Length of Fwd Packet": pd.Series([], dtype=float),
            "Packet Length Mean": pd.Series([], dtype=float),
        }
    )
    with pytest.raises(ValueError, match=r"no \(src, dst\) channel reaches min_events"):
        channel_dispersion(empty, min_events=1)


def test_non_numeric_bytes_score_zero_instead_of_nan() -> None:
    # Corrupt/non-numeric payload columns coerce to all-NaN; _cv must survive
    # the "Mean of empty slice" nanmean/nanstd case rather than propagating NaN.
    n = 16
    flows = pd.DataFrame(
        {
            "Src IP": ["10.0.0.9"] * n,
            "Dst IP": ["203.0.113.99"] * n,
            "Timestamp": [f"07/07/2017 05:{m:02d}:00 PM" for m in range(n)],
            "Total Length of Fwd Packet": ["garbage"] * n,
            "Packet Length Mean": ["also-garbage"] * n,
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        channels = channel_dispersion(flows, min_events=16)

    assert any("Mean of empty slice" in str(w.message) for w in caught)
    assert channels.loc[0, "fwd_bytes_cv"] == 0.0
    assert channels.loc[0, "pkt_len_cv"] == 0.0


def test_single_flow_channel_scores_zero_dispersion() -> None:
    single = _channel("10.0.0.10", "192.168.1.10", [123], start_hour=6)
    channels = channel_dispersion(single, min_events=1)
    assert channels.loc[0, "n_events"] == 1
    assert channels.loc[0, "fwd_bytes_cv"] == 0.0


def test_scorer_falls_back_to_unit_iqr_when_benign_stats_are_constant() -> None:
    # Every benign channel scores identically -> IQR is 0 for both stats; the
    # scorer must fall back to an IQR of 1.0 instead of dividing by zero.
    benign_stats = np.array([[0.1, 0.2]] * 5)
    scorer = BeaconScorer().fit(benign_stats)

    assert scorer.iqr is not None
    assert (scorer.iqr == 1.0).all()

    scores = scorer.score(np.array([[0.1, 0.2], [5.0, 5.0]]))
    assert scores[0] == 0.0
    assert scores[1] > 0.0


def test_main_detects_beacon_channel_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full CLI path on a tiny synthetic corpus: an ARES-style poll+tasking
    channel must be flagged while uniform benign channels stay clean."""
    from sentinel.ids.beacon import main

    monkeypatch.setenv("SENTINEL_MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    rng = np.random.default_rng(7)
    data_dir = tmp_path / "cicids"
    data_dir.mkdir()

    def _uniform_channel(src: str, dst: str, start_hour: int) -> pd.DataFrame:
        sizes = list(np.round(76 + rng.normal(0, 1, 20)).astype(int))
        return _channel(src, dst, sizes, start_hour=start_hour)

    # Train days (Monday/Tuesday are in TRAIN_DAYS): benign channels only.
    train = pd.concat(
        [
            _uniform_channel(f"10.0.0.{i + 1}", f"192.168.1.{i + 1}", start_hour=1 + i)
            for i in range(3)
        ],
        ignore_index=True,
    )
    train.to_csv(data_dir / "Monday-WorkingHours.csv", index=False)

    # Test day (Thursday, not in TRAIN_DAYS): benign channels + an ARES-style
    # beacon interleaving empty "- Attempted" polls with data-carrying tasking.
    test_frames = [
        _uniform_channel(f"10.0.1.{i + 1}", f"192.168.2.{i + 1}", start_hour=1 + i)
        for i in range(2)
    ]
    beacon_sizes: list[int] = []
    beacon_labels: list[str] = []
    for i in range(20):
        if i % 2 == 0:
            beacon_sizes.append(0)
            beacon_labels.append("Bot - Attempted")
        else:
            beacon_sizes.append(int(3000 + rng.integers(0, 3000)))
            beacon_labels.append("Bot")
    test_frames.append(
        _channel("10.0.5.5", "203.0.113.9", beacon_sizes, start_hour=5, label=beacon_labels)
    )
    pd.concat(test_frames, ignore_index=True).to_csv(
        data_dir / "Thursday-WorkingHours.csv", index=False
    )

    metrics = main(["--data-dir", str(data_dir), "--min-events", "16"])

    assert metrics["false_positive_rate"] == 0.0
    assert metrics["recall_overall"] == 1.0
    assert metrics["recall__Bot"] == 1.0
