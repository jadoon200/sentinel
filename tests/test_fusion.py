"""Fusion scoring: a shared tag is not a match — rarity, recency and corroboration are."""

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sentinel.correlate.fusion import (
    _corroboration_factor,
    _recency_factor,
    _soft_or,
    score_campaign_matches,
    technique_idf,
)
from sentinel.db.base import Base
from sentinel.db.models import (
    AttackTechnique,
    Campaign,
    CampaignReport,
    CampaignTechnique,
    ReportTechnique,
    ThreatReport,
)

NOW = datetime(2026, 6, 13)


def _component_tests() -> None:
    # soft-or stays bounded and compounds weak evidence
    assert _soft_or([]) == 0.0
    assert _soft_or([1.0]) == 1.0
    assert abs(_soft_or([0.5, 0.5]) - 0.75) < 1e-9
    # corroboration saturates with report count
    assert _corroboration_factor(0.8, 1) < _corroboration_factor(0.8, 3)
    assert _corroboration_factor(0.8, 3) < 0.8
    # recency: fresh ~1, half-life -> 0.5, unknown -> neutral
    assert _recency_factor(0.0, 30.0) == 1.0
    assert abs(_recency_factor(30.0, 30.0) - 0.5) < 1e-9
    assert _recency_factor(None, 30.0) == 1.0


def test_component_helpers_are_bounded_and_monotonic() -> None:
    _component_tests()


def _seed(session: Session) -> None:
    """Two campaigns sharing a technique with an alert.

    `camp:rare` shares T1195.001 (supply-chain — appears in 1 of 5 corpus reports,
    so it is rare/surprising) and was reported today.
    `camp:common` shares T1110 (brute force — appears in every corpus report, so it
    is generic) and was last reported four months ago.
    A reviewer's exact objection: both "match" on a shared tag, but only the first
    is meaningful. Fusion strength must separate them.
    """
    for tid, name in [
        ("T1195.001", "Supply Chain Compromise: Compromise Software Dependencies"),
        ("T1110", "Brute Force"),
    ]:
        session.add(AttackTechnique(technique_id=tid, name=name))

    # Corpus rarity: T1110 in all 5 reports, T1195.001 in just one.
    for i in range(5):
        rid = f"rss:{i}"
        session.add(ThreatReport(report_id=rid, source="rss", title="x", ingested_at=NOW))
        session.add(
            ReportTechnique(
                report_id=rid, technique_id="T1110", score=0.5, corroborations=1, method="t"
            )
        )
    session.add(
        ReportTechnique(
            report_id="rss:0", technique_id="T1195.001", score=0.5, corroborations=1, method="t"
        )
    )

    # Rare + recent + well-corroborated campaign.
    session.add(Campaign(campaign_id="camp:rare", cve_ids=["CVE-2026-1"], report_count=3))
    session.add(CampaignReport(campaign_id="camp:rare", report_id="rss:0"))
    session.add(ThreatReport(report_id="rss:0r", source="rss", title="fresh", published=NOW))
    session.add(CampaignReport(campaign_id="camp:rare", report_id="rss:0r"))
    session.add(
        CampaignTechnique(
            campaign_id="camp:rare",
            technique_id="T1195.001",
            corroborations=3,
            score=0.8,
            method="cve-component-fusion",
        )
    )

    # Common + stale + weakly-corroborated campaign.
    session.add(Campaign(campaign_id="camp:common", cve_ids=["CVE-2026-2"], report_count=2))
    session.add(
        ThreatReport(
            report_id="rss:old", source="rss", title="stale", published=NOW - timedelta(days=120)
        )
    )
    session.add(CampaignReport(campaign_id="camp:common", report_id="rss:old"))
    session.add(
        CampaignTechnique(
            campaign_id="camp:common",
            technique_id="T1110",
            corroborations=1,
            score=0.4,
            method="cve-component-fusion",
        )
    )
    session.commit()


def test_idf_ranks_rare_technique_above_common() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed(session)
        idf = technique_idf(session)
    # T1195.001 (1/5 reports) is more surprising than T1110 (5/5 reports).
    assert idf["T1195.001"] > idf["T1110"]
    assert idf["T1110"] == 0.0  # the most common technique scores zero rarity
    assert idf["T1195.001"] == 1.0


def test_specific_recent_match_outranks_generic_stale_one() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed(session)
        matches = score_campaign_matches(session, {"T1195.001", "T1110"}, now=NOW)

    assert [m.campaign_id for m in matches] == ["camp:rare", "camp:common"]
    rare, common = matches
    # The headline number reflects the gap, and every component explains why.
    assert rare.fusion.strength > common.fusion.strength
    assert rare.fusion.specificity > common.fusion.specificity  # rarer tag
    assert rare.fusion.recency > common.fusion.recency  # active vs months old
    assert rare.fusion.corroboration > common.fusion.corroboration  # more reports
    assert 0.0 <= common.fusion.strength <= rare.fusion.strength <= 1.0
    assert rare.fusion.age_days is not None and rare.fusion.age_days < 1.0


def test_no_overlap_yields_no_matches() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed(session)
        assert score_campaign_matches(session, set(), now=NOW) == []
        assert score_campaign_matches(session, {"T9999"}, now=NOW) == []


def test_parent_alert_matches_subtechnique_campaign() -> None:
    """A DoS alert tagged the parent T1499 must fuse with a campaign tagged the
    sub-technique T1499.004 — the IDS map emits parents, the NLP tagger subs."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            AttackTechnique(technique_id="T1499.004", name="Application or System Exploitation")
        )
        session.add(ThreatReport(report_id="rss:dos", source="rss", title="DoS", published=NOW))
        session.add(
            ReportTechnique(
                report_id="rss:dos",
                technique_id="T1499.004",
                score=0.6,
                corroborations=2,
                method="t",
            )
        )
        session.add(Campaign(campaign_id="camp:dos", cve_ids=["CVE-2026-9"], report_count=2))
        session.add(CampaignReport(campaign_id="camp:dos", report_id="rss:dos"))
        session.add(
            CampaignTechnique(
                campaign_id="camp:dos",
                technique_id="T1499.004",
                corroborations=2,
                score=0.6,
                method="cve-component-fusion",
            )
        )
        session.commit()

        # Parent alert tag matches the sub-technique campaign tag.
        parent = score_campaign_matches(session, {"T1499"}, now=NOW)
        assert [m.campaign_id for m in parent] == ["camp:dos"]
        assert parent[0].matched_techniques == ["T1499.004"]  # honest: reports the sub
        # Exact sub-technique tag still matches; an unrelated family does not.
        assert score_campaign_matches(session, {"T1499.004"}, now=NOW)[0].campaign_id == "camp:dos"
        assert score_campaign_matches(session, {"T1498"}, now=NOW) == []
