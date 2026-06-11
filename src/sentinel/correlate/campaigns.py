"""Group threat reports into campaigns and fuse their technique evidence.

A campaign is a connected component of reports linked by shared CVE mentions —
deliberately simple and explainable. Technique evidence from member reports is
re-aggregated with the same corroboration logic used at sentence level, so the
campaign view benefits from multi-report aggregation (arXiv:2604.07470).

Campaigns are derived artifacts: every run rebuilds them from report_cves and
report_techniques, so they stay consistent as new reports arrive.
"""

import hashlib

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sentinel.correlate.cves import extract_cve_ids
from sentinel.db.models import (
    AttackTechnique,
    Campaign,
    CampaignReport,
    CampaignTechnique,
    ReportCve,
    ReportTechnique,
    ThreatReport,
)
from sentinel.nlp.mapper import TechniqueMatch, aggregate_matches


def link_report_cves(session: Session) -> int:
    """Upsert report→CVE mention edges for all reports (idempotent)."""
    edges = 0
    for report in session.scalars(select(ThreatReport)):
        text = " ".join(part for part in (report.title, report.summary) if part)
        for cve_id in extract_cve_ids(text):
            session.merge(ReportCve(report_id=report.report_id, cve_id=cve_id))
            edges += 1
    return edges


def _find(parent: dict[str, str], node: str) -> str:
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:  # path compression
        parent[node], node = root, parent[node]
    return root


def _union(parent: dict[str, str], a: str, b: str) -> None:
    parent[_find(parent, a)] = _find(parent, b)


def _campaign_id(cve_ids: list[str]) -> str:
    digest = hashlib.sha256("|".join(sorted(cve_ids)).encode()).hexdigest()[:16]
    return f"camp:{digest}"


def build_campaigns(session: Session, min_reports: int = 2) -> int:
    """Rebuild campaign tables from CVE-mention components. Returns campaign count."""
    rows = session.execute(select(ReportCve.report_id, ReportCve.cve_id)).all()

    parent = {report_id: report_id for report_id, _ in rows}
    by_cve: dict[str, list[str]] = {}
    for report_id, cve_id in rows:
        by_cve.setdefault(cve_id, []).append(report_id)
    for cve_reports in by_cve.values():
        for other in cve_reports[1:]:
            _union(parent, cve_reports[0], other)

    components: dict[str, set[str]] = {}
    for report_id, _ in rows:
        components.setdefault(_find(parent, report_id), set()).add(report_id)
    campaigns = [members for members in components.values() if len(members) >= min_reports]

    # Derived tables are rebuilt wholesale each run.
    session.execute(delete(CampaignTechnique))
    session.execute(delete(CampaignReport))
    session.execute(delete(Campaign))

    cves_by_report: dict[str, list[str]] = {}
    for report_id, cve_id in rows:
        cves_by_report.setdefault(report_id, []).append(cve_id)
    names = {
        technique_id: name
        for technique_id, name in session.execute(
            select(AttackTechnique.technique_id, AttackTechnique.name)
        )
    }

    for members in campaigns:
        cve_ids = sorted({cve for r in members for cve in cves_by_report[r]})
        campaign_id = _campaign_id(cve_ids)
        session.add(Campaign(campaign_id=campaign_id, cve_ids=cve_ids, report_count=len(members)))
        per_report_matches = []
        for report_id in sorted(members):
            session.add(CampaignReport(campaign_id=campaign_id, report_id=report_id))
            edges = session.scalars(
                select(ReportTechnique).where(ReportTechnique.report_id == report_id)
            ).all()
            if edges:
                per_report_matches.append(
                    [
                        TechniqueMatch(e.technique_id, names.get(e.technique_id, ""), e.score)
                        for e in edges
                    ]
                )
        for match in aggregate_matches(per_report_matches):
            session.add(
                CampaignTechnique(
                    campaign_id=campaign_id,
                    technique_id=match.technique_id,
                    corroborations=match.corroborations,
                    score=match.score,
                    method="cve-component-fusion",
                )
            )
    return len(campaigns)
