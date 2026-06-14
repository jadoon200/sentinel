"""Beacon-by-dispersion: a poll+tasking channel out-scores a uniform benign one."""

import pandas as pd

from sentinel.ids.beacon import BeaconScorer, channel_dispersion


def _channel(src: str, dst: str, byte_sizes: list[int], start_hour: int) -> pd.DataFrame:
    n = len(byte_sizes)
    ts = [f"07/07/2017 {start_hour:02d}:{m:02d}:00 PM" for m in range(n)]
    return pd.DataFrame(
        {
            "Src IP": [src] * n,
            "Dst IP": [dst] * n,
            "Timestamp": ts,
            "Total Length of Fwd Packet": byte_sizes,
            "Packet Length Mean": [b / 2.0 for b in byte_sizes],
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
