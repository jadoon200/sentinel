"""WAF replay: score HTTP requests through the SQLi detector into fused alerts.

The flow replay (`ids/replay.py`) turns network flows into ATT&CK-tagged alerts;
this does the same for the application layer, so SQLi is a first-class signal in
the platform — not just a standalone detector. It trains the payload SQLi model
on one corpus, scores a held-out *different* corpus as an incoming request stream
(cross-corpus, the honest setting), and persists each flagged request as a
`model="sqli"` Alert tagged T1190, which then fuses with T1190 campaigns and
surfaces in the host rollup exactly like a flow detection.

Honest framing: the public payload corpora carry no client IPs, so — like the
flow replay over the CIC-IDS2017 *testbed* — this is a replay over a labelled
dataset, not production capture. Requests are attributed to synthetic client IPs
from the RFC 5737 documentation range (203.0.113.0/24) so the per-host rollup has
something to group on; those IPs are deliberately the reserved "example" block,
signalling they are not real attribution.

Usage (Postgres up via `make up`): python -m sentinel.ids.waf_replay
"""

import argparse

import numpy as np

from sentinel.ids.sqli import build_detector, load_corpora

# RFC 5737 TEST-NET-3 — reserved for documentation, never routable: an explicit
# "synthetic" marker for the demo client attribution.
_SYNTHETIC_CLIENTS = [f"203.0.113.{i}" for i in range(1, 9)]


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-corpus", default="httpparams")
    parser.add_argument("--stream-corpus", default="sqliv2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-alerts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    from sqlalchemy import delete

    from sentinel.db.base import session_scope
    from sentinel.db.models import Alert

    corpora = load_corpora()
    train_texts, train_labels = corpora[args.train_corpus]
    stream_texts, stream_labels = corpora[args.stream_corpus]

    detector = build_detector().fit(train_texts, train_labels)
    proba = np.asarray(detector.predict_proba(stream_texts))[:, 1]
    flagged = np.flatnonzero(proba >= args.threshold)
    # Highest-confidence detections first, capped — the dashboard's top alerts.
    flagged = flagged[np.argsort(-proba[flagged])][: args.max_alerts]

    alerts = [
        Alert(
            model="sqli",
            day=None,
            score=float(proba[i]),
            predicted_label="sqli",
            true_label="Web Attack - Sql Injection" if stream_labels[i] == 1 else "BENIGN",
            techniques=["T1190"],  # Exploit Public-Facing Application
            source_host=_SYNTHETIC_CLIENTS[int(i) % len(_SYNTHETIC_CLIENTS)],
        )
        for i in flagged
    ]

    with session_scope() as session:
        # Idempotent and non-clobbering: only the WAF's own rows are rebuilt, so a
        # WAF replay and a flow replay coexist in the alerts table.
        session.execute(delete(Alert).where(Alert.model == "sqli"))
        for alert in alerts:
            session.add(alert)

    true_positives = sum(1 for a in alerts if a.true_label != "BENIGN")
    counts = {"sqli_alerts": len(alerts), "true_sqli": true_positives}
    print(counts)
    return counts


if __name__ == "__main__":
    main()
