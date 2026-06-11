import numpy as np
import pandas as pd

from sentinel.ids.profile import ProfileScorer, build_window_stats


def _flows(host: str, n: int, ports: list[int], start_min: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Src IP": [host] * n,
            "Dst IP": ["192.168.10.50"] * n,
            "Dst Port": [ports[i % len(ports)] for i in range(n)],
            "Timestamp": [
                f"03/07/2017 01:{start_min + i // 60:02d}:{i % 60:02d} PM" for i in range(n)
            ],
            "Total Fwd Packet": [10] * n,
        }
    )


def test_scanner_fanout_scores_above_benign() -> None:
    # benign: one host reusing two services; scanner: distinct port per flow
    benign = _flows("10.0.0.1", 64, [80, 443])
    scanner = _flows("10.0.0.9", 64, list(range(1, 65)))

    benign_stats, _ = build_window_stats(benign, window=16, stride=8)
    scan_stats, last_pos = build_window_stats(scanner, window=16, stride=8)

    scorer = ProfileScorer().fit(benign_stats)
    threshold = np.percentile(scorer.score(benign_stats), 99)

    assert (scorer.score(scan_stats) > threshold).all()
    assert last_pos[0] == 15  # window labeled by its last flow's position
