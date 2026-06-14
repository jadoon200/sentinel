"""Payload SQLi detector: char n-grams separate injection strings from benign input."""

import numpy as np

from sentinel.ids.sqli import TECHNIQUES, build_detector


def test_detector_flags_sqli_over_benign() -> None:
    sqli = [
        "1' OR '1'='1",
        "admin'--",
        "1; DROP TABLE users;--",
        "' UNION SELECT username, password FROM users--",
        "1) or sleep(5)#",
        "%27 or 1=1 --",
    ]
    benign = [
        "london",
        "user@example.com",
        "2026-06-14",
        "blue widget size large",
        "order_id=10482",
        "Jayden",
    ]
    texts = sqli + benign
    labels = np.array([1] * len(sqli) + [0] * len(benign))

    detector = build_detector().fit(texts, labels)
    proba = np.asarray(detector.predict_proba(texts))[:, 1]

    # Every SQLi payload outscores every benign one (clean separation on this set).
    assert proba[: len(sqli)].min() > proba[len(sqli) :].max()
    assert TECHNIQUES == ["T1190"]  # maps into the same ATT&CK graph
