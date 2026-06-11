from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from sentinel.correlate.campaigns import build_campaigns, link_report_cves
from sentinel.correlate.cves import extract_cve_ids
from sentinel.db.base import Base
from sentinel.db.models import (
    AttackTechnique,
    Campaign,
    CampaignTechnique,
    ReportTechnique,
    ThreatReport,
)


def test_extract_cve_ids_dedupes_and_normalizes() -> None:
    text = "Exploits cve-2026-12345 and CVE-2026-12345; also CVE-2025-0001. Not CVE-99."
    assert extract_cve_ids(text) == ["CVE-2026-12345", "CVE-2025-0001"]


def _seed(session: Session) -> None:
    session.add(AttackTechnique(technique_id="T1190", name="Exploit Public-Facing Application"))
    session.add(AttackTechnique(technique_id="T1059", name="Command and Scripting Interpreter"))
    reports = [
        ("rss:a", "Actors exploit CVE-2026-1111 in VPN appliances"),
        ("rss:b", "Follow-up: CVE-2026-1111 now used to drop loaders"),
        ("rss:c", "Unrelated advisory for CVE-2026-9999"),
    ]
    for report_id, title in reports:
        session.add(ThreatReport(report_id=report_id, source="rss", title=title))
    session.add(
        ReportTechnique(
            report_id="rss:a", technique_id="T1190", score=0.5, corroborations=3, method="t"
        )
    )
    session.add(
        ReportTechnique(
            report_id="rss:b", technique_id="T1190", score=0.4, corroborations=2, method="t"
        )
    )
    session.add(
        ReportTechnique(
            report_id="rss:b", technique_id="T1059", score=0.6, corroborations=1, method="t"
        )
    )
    session.commit()


def test_build_campaigns_clusters_by_shared_cve_and_fuses_techniques() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        _seed(session)

        assert link_report_cves(session) == 3
        assert build_campaigns(session) == 1
        session.commit()

        campaign = session.scalars(select(Campaign)).one()
        assert campaign.cve_ids == ["CVE-2026-1111"]
        assert campaign.report_count == 2

        fused = session.scalars(
            select(CampaignTechnique).order_by(CampaignTechnique.corroborations.desc())
        ).all()
        assert [(f.technique_id, f.corroborations) for f in fused] == [("T1190", 2), ("T1059", 1)]

        # Rebuild is idempotent: same campaign id, no duplicate rows.
        first_id = campaign.campaign_id
        assert build_campaigns(session) == 1
        session.commit()
        assert session.scalars(select(Campaign)).one().campaign_id == first_id
