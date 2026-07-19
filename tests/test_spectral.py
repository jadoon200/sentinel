import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.ids.spectral import default_period_grid, pair_scores, schuster_score


def test_schuster_separates_timer_from_poisson() -> None:
    rng = np.random.default_rng(13)
    periods = default_period_grid()
    # Tick-based beacon (bounded jitter) and sleep-based beacon (drifting
    # jitter accumulates as a random walk) vs a memoryless Poisson stream.
    tick_beacon = 60.0 * np.arange(64) + rng.uniform(-3, 3, 64)
    sleep_beacon = np.cumsum(rng.normal(60, 3, 64))
    poisson = np.cumsum(rng.exponential(60, 64))

    noise_floor = schuster_score(poisson, periods)
    assert schuster_score(tick_beacon, periods) > 2.0 * noise_floor
    assert schuster_score(sleep_beacon, periods) > 1.5 * noise_floor


def _pair_frame(dst: str, times: np.ndarray, label: str = "BENIGN") -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "Src IP": ["10.0.0.5"] * n,
            "Dst IP": [dst] * n,
            "Timestamp": [
                f"03/07/2017 {1 + int(t) // 3600:02d}:"
                f"{(int(t) // 60) % 60:02d}:{int(t) % 60:02d} PM"
                for t in times
            ],
            "Label": [label] * n,
        }
    )


def test_pair_scores_isolate_the_beacon_channel() -> None:
    rng = np.random.default_rng(13)
    beacon = _pair_frame("205.174.165.73", 60.0 * np.arange(32) + rng.uniform(-2, 2, 32))
    chatter = _pair_frame("192.168.10.50", np.cumsum(rng.exponential(45, 32)))
    flows = pd.concat([beacon, chatter], ignore_index=True)

    scores = pair_scores(flows, min_events=16)

    by_dst = scores.set_index("dst")["score"]
    assert by_dst["205.174.165.73"] > 2 * by_dst["192.168.10.50"]
    assert (scores["n_events"] == 32).all()


def test_schuster_score_survives_series_shorter_than_window() -> None:
    # Fewer events than the analysis window means no window ever fills, so
    # the function must fall back to 0.0 instead of indexing past the array.
    short = np.array([0.0, 5.0, 9.0, 14.0, 20.0])
    assert schuster_score(short, default_period_grid(), window=16) == 0.0


def test_schuster_score_survives_burst_too_tight_for_any_period() -> None:
    # A burst of events all within a couple of seconds has a window span
    # below the period grid's floor (4s) -> every period is invalid.
    tight_burst = np.cumsum(np.full(20, 0.1))
    assert schuster_score(tight_burst, default_period_grid(), window=16) == 0.0


def test_schuster_score_survives_constant_timestamps() -> None:
    # All events at the same instant: zero span, so (like the tight-burst
    # case) no period in the grid is valid — must not crash or divide by zero.
    constant = np.zeros(20)
    score = schuster_score(constant, default_period_grid())
    assert score == 0.0


def test_schuster_score_empty_and_single_event_return_zero() -> None:
    periods = default_period_grid()
    assert schuster_score(np.array([]), periods) == 0.0
    assert schuster_score(np.array([0.0]), periods) == 0.0


def test_schuster_score_reproduces_mean_of_empty_slice_but_stays_finite() -> None:
    # Short Poisson-ish channels leave some period columns valid in only a
    # subset of windows, so nanmean legitimately hits an all-NaN slice for
    # the rest. The known warning is expected; the returned score must not be.
    rng = np.random.default_rng(11)
    seconds = np.cumsum(rng.exponential(45, 32))
    periods = default_period_grid()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = schuster_score(seconds, periods)

    assert any("Mean of empty slice" in str(w.message) for w in caught)
    assert np.isfinite(score)


def test_min_events_filters_out_short_channels() -> None:
    short = _pair_frame("192.168.9.1", np.arange(5, dtype=float))
    long_enough = _pair_frame("192.168.9.2", np.arange(16, dtype=float))
    flows = pd.concat([short, long_enough], ignore_index=True)

    channels = pair_scores(flows, min_events=16)

    assert len(channels) == 1
    assert channels.iloc[0]["dst"] == "192.168.9.2"


def test_no_channel_reaching_min_events_raises() -> None:
    short = _pair_frame("192.168.9.1", np.arange(5, dtype=float))
    with pytest.raises(ValueError, match=r"no \(src, dst\) channel reaches min_events"):
        pair_scores(short, min_events=16)


def test_empty_frame_raises_value_error() -> None:
    empty = pd.DataFrame(
        {
            "Src IP": pd.Series([], dtype=str),
            "Dst IP": pd.Series([], dtype=str),
            "Timestamp": pd.Series([], dtype=str),
        }
    )
    with pytest.raises(ValueError, match=r"no \(src, dst\) channel reaches min_events"):
        pair_scores(empty, min_events=1)


def test_single_event_channel_scores_zero() -> None:
    single = _pair_frame("192.168.1.10", np.array([0.0]))
    channels = pair_scores(single, min_events=1)
    assert channels.loc[0, "n_events"] == 1
    assert channels.loc[0, "score"] == 0.0


def test_main_detects_periodic_beacon_channel_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full CLI path on a tiny synthetic corpus: a fixed-period channel must
    be flagged while a Poisson-ish benign channel stays clean."""
    from sentinel.ids.spectral import main

    monkeypatch.setenv("SENTINEL_MLFLOW_TRACKING_URI", f"file:{tmp_path}/mlruns")
    rng = np.random.default_rng(11)
    data_dir = tmp_path / "cicids"
    data_dir.mkdir()

    def _benign_channel(dst: str) -> pd.DataFrame:
        seconds = np.cumsum(rng.exponential(45, 32))
        return _pair_frame(dst, seconds)

    # Train days (Monday is in TRAIN_DAYS): benign, non-periodic channels only.
    train = pd.concat([_benign_channel("192.168.1.1"), _benign_channel("192.168.1.2")])
    train.to_csv(data_dir / "Monday-WorkingHours.csv", index=False)

    # Test day (Thursday, not in TRAIN_DAYS): one benign channel plus a
    # tightly-periodic C2 beacon (fixed 60s cadence, small jitter).
    beacon_seconds = 60.0 * np.arange(32) + rng.uniform(-2, 2, 32)
    test = pd.concat(
        [_benign_channel("192.168.2.1"), _pair_frame("205.174.165.73", beacon_seconds, label="Bot")]
    )
    test.to_csv(data_dir / "Thursday-WorkingHours.csv", index=False)

    metrics = main(["--data-dir", str(data_dir), "--min-events", "16"])

    assert metrics["false_positive_rate"] == 0.0
    assert metrics["recall_overall"] == 1.0
    assert metrics["recall__Bot"] == 1.0
