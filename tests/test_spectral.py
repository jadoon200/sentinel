import numpy as np
import pandas as pd

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


def _pair_frame(dst: str, times: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Src IP": ["10.0.0.5"] * len(times),
            "Dst IP": [dst] * len(times),
            "Timestamp": [
                f"03/07/2017 {1 + int(t) // 3600:02d}:"
                f"{(int(t) // 60) % 60:02d}:{int(t) % 60:02d} PM"
                for t in times
            ],
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
