from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sentinel.correlate.trending import briefing_lines, feed_drift, trending_techniques
from sentinel.db.base import Base
from sentinel.db.models import AttackTechnique, ReportTechnique, ThreatReport


def _seed(session: Session, now: datetime) -> None:
    session.add(AttackTechnique(technique_id="T1190", name="Exploit Public-Facing App"))
    session.add(AttackTechnique(technique_id="T1566", name="Phishing"))
    # T1190 surges this week; T1566 steady; sources shift otx -> rss.
    rows = [
        ("r1", "otx", now - timedelta(days=1), "T1190"),
        ("r2", "rss", now - timedelta(days=2), "T1190"),
        ("r3", "rss", now - timedelta(days=3), "T1190"),
        ("r4", "rss", now - timedelta(days=10), "T1566"),
        ("r5", "otx", now - timedelta(days=11), "T1566"),
        ("r6", "otx", now - timedelta(days=12), "T1190"),
    ]
    for rid, src, ts, tech in rows:
        session.add(ThreatReport(report_id=rid, source=src, title=rid, ingested_at=ts))
        session.add(
            ReportTechnique(
                report_id=rid, technique_id=tech, score=0.5, corroborations=1, method="t"
            )
        )
    session.commit()


def test_trending_surfaces_the_surging_technique() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    now = datetime(2026, 6, 12).astimezone()
    with Session(engine) as session:
        _seed(session, now)
        trending = trending_techniques(session, now=now, window_days=7)

    assert trending[0].technique_id == "T1190"
    assert trending[0].recent_count == 3 and trending[0].prior_count == 1


def test_feed_drift_flags_source_shift() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    now = datetime(2026, 6, 12).astimezone()
    with Session(engine) as session:
        _seed(session, now)
        drift = feed_drift(session, now=now, window_days=7)

    # recent window is rss-heavy, prior was otx-heavy -> non-zero PSI
    assert drift.population_stability_index > 0.0
    assert drift.top_shifts[0][0] in {"rss", "otx"}


def test_briefing_lines_render() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    now = datetime(2026, 6, 12).astimezone()
    with Session(engine) as session:
        _seed(session, now)
        trending = trending_techniques(session, now=now)
        drift = feed_drift(session, now=now)
        lines = briefing_lines(trending, drift, n_campaigns=2, n_kev=1)

    text = "\n".join(lines)
    assert "daily threat briefing" in text
    assert "T1190" in text
