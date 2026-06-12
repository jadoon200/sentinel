"""Read-only API over the SENTINEL knowledge graph.

Serves the fused threat picture: campaigns with corroborated techniques,
tagged reports, IDS alerts, and per-technique evidence across both layers.
Run locally: `make api` (http://localhost:8000/docs).
"""

from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sentinel.db.base import get_session_factory
from sentinel.db.models import (
    Alert,
    AttackTechnique,
    Campaign,
    CampaignReport,
    CampaignTechnique,
    KevEntry,
    ReportTechnique,
    ThreatReport,
    Vulnerability,
)

app = FastAPI(
    title="SENTINEL",
    description="Cyber threat intelligence fusion: OSINT + NLP + IDS in one ATT&CK graph",
    version="0.1.0",
)

# Read-only API; the Vite dev server and any local dashboard build may call it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]


class Stats(BaseModel):
    vulnerabilities: int
    kev_entries: int
    attack_techniques: int
    threat_reports: int
    report_technique_edges: int
    campaigns: int
    alerts: int
    alerts_by_model: dict[str, int]


class TechniqueEvidence(BaseModel):
    technique_id: str
    name: str | None
    score: float
    corroborations: int


class CampaignSummary(BaseModel):
    campaign_id: str
    cve_ids: list[str]
    # CVEs on the CISA Known Exploited Vulnerabilities catalog — the campaign
    # involves vulnerabilities with confirmed exploitation in the wild.
    kev_cves: list[str]
    report_count: int
    techniques: list[TechniqueEvidence]


class ReportSummary(BaseModel):
    report_id: str
    source: str
    title: str
    url: str | None
    published: datetime | None
    techniques: list[TechniqueEvidence]


class CampaignDetail(CampaignSummary):
    reports: list[ReportSummary]


class AlertOut(BaseModel):
    alert_id: int
    model: str
    day: str | None
    score: float
    predicted_label: str | None
    true_label: str | None
    techniques: list[str]


class CampaignMatch(BaseModel):
    campaign_id: str
    cve_ids: list[str]
    kev_cves: list[str]
    report_count: int
    matched_techniques: list[str]


class AlertContext(BaseModel):
    alert: "AlertOut"
    matched_campaigns: list[CampaignMatch]


class TechniqueDetail(BaseModel):
    technique_id: str
    name: str
    tactics: list[str]
    url: str | None
    report_count: int
    campaign_count: int
    alert_count: int


def _technique_names(session: Session, technique_ids: set[str]) -> dict[str, str]:
    if not technique_ids:
        return {}
    rows = session.execute(
        select(AttackTechnique.technique_id, AttackTechnique.name).where(
            AttackTechnique.technique_id.in_(technique_ids)
        )
    )
    return {technique_id: name for technique_id, name in rows}


def _kev_overlap(session: Session, cve_ids: list[str]) -> list[str]:
    if not cve_ids:
        return []
    rows = session.scalars(select(KevEntry.cve_id).where(KevEntry.cve_id.in_(cve_ids)))
    return sorted(rows)


def _campaign_techniques(session: Session, campaign_id: str) -> list[TechniqueEvidence]:
    edges = session.scalars(
        select(CampaignTechnique)
        .where(CampaignTechnique.campaign_id == campaign_id)
        .order_by(CampaignTechnique.corroborations.desc(), CampaignTechnique.score.desc())
    ).all()
    names = _technique_names(session, {e.technique_id for e in edges})
    return [
        TechniqueEvidence(
            technique_id=e.technique_id,
            name=names.get(e.technique_id),
            score=e.score,
            corroborations=e.corroborations,
        )
        for e in edges
    ]


def _report_summary(session: Session, report: ThreatReport) -> ReportSummary:
    edges = session.scalars(
        select(ReportTechnique)
        .where(ReportTechnique.report_id == report.report_id)
        .order_by(ReportTechnique.corroborations.desc(), ReportTechnique.score.desc())
    ).all()
    names = _technique_names(session, {e.technique_id for e in edges})
    return ReportSummary(
        report_id=report.report_id,
        source=report.source,
        title=report.title,
        url=report.url,
        published=report.published,
        techniques=[
            TechniqueEvidence(
                technique_id=e.technique_id,
                name=names.get(e.technique_id),
                score=e.score,
                corroborations=e.corroborations,
            )
            for e in edges
        ],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats(session: SessionDep) -> Stats:
    def count(model: type) -> int:
        return session.scalar(select(func.count()).select_from(model)) or 0

    return Stats(
        vulnerabilities=count(Vulnerability),
        kev_entries=count(KevEntry),
        attack_techniques=count(AttackTechnique),
        threat_reports=count(ThreatReport),
        report_technique_edges=count(ReportTechnique),
        campaigns=count(Campaign),
        alerts=count(Alert),
        alerts_by_model={
            model: n
            for model, n in session.execute(select(Alert.model, func.count()).group_by(Alert.model))
        },
    )


@app.get("/campaigns")
def list_campaigns(session: SessionDep) -> list[CampaignSummary]:
    campaigns = session.scalars(select(Campaign).order_by(Campaign.report_count.desc())).all()
    return [
        CampaignSummary(
            campaign_id=c.campaign_id,
            cve_ids=c.cve_ids,
            kev_cves=_kev_overlap(session, c.cve_ids),
            report_count=c.report_count,
            techniques=_campaign_techniques(session, c.campaign_id),
        )
        for c in campaigns
    ]


@app.get("/campaigns/{campaign_id}")
def campaign_detail(campaign_id: str, session: SessionDep) -> CampaignDetail:
    campaign = session.get(Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    member_ids = session.scalars(
        select(CampaignReport.report_id).where(CampaignReport.campaign_id == campaign_id)
    ).all()
    reports = session.scalars(
        select(ThreatReport).where(ThreatReport.report_id.in_(member_ids))
    ).all()
    return CampaignDetail(
        campaign_id=campaign.campaign_id,
        cve_ids=campaign.cve_ids,
        kev_cves=_kev_overlap(session, campaign.cve_ids),
        report_count=campaign.report_count,
        techniques=_campaign_techniques(session, campaign_id),
        reports=[_report_summary(session, r) for r in reports],
    )


@app.get("/reports")
def list_reports(
    session: SessionDep,
    source: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ReportSummary]:
    query = (
        select(ThreatReport)
        .order_by(ThreatReport.ingested_at.desc())
        .limit(min(limit, 200))
        .offset(max(offset, 0))
    )
    if source is not None:
        query = query.where(ThreatReport.source == source)
    return [_report_summary(session, r) for r in session.scalars(query).all()]


@app.get("/alerts")
def list_alerts(
    session: SessionDep,
    model: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[AlertOut]:
    query = select(Alert).order_by(Alert.score.desc()).limit(min(limit, 200)).offset(max(offset, 0))
    if model is not None:
        query = query.where(Alert.model == model)
    return [
        AlertOut(
            alert_id=a.alert_id,
            model=a.model,
            day=a.day,
            score=a.score,
            predicted_label=a.predicted_label,
            true_label=a.true_label,
            techniques=a.techniques or [],
        )
        for a in session.scalars(query).all()
    ]


@app.get("/alerts/{alert_id}/context")
def alert_context(alert_id: int, session: SessionDep) -> AlertContext:
    """Fusion join: campaigns whose technique evidence overlaps this alert's techniques.

    This is the platform's core correlation — an IDS detection gains threat-intel
    context ("these techniques are active in campaign X reported last week").
    """
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="alert not found")

    alert_out = AlertOut(
        alert_id=alert.alert_id,
        model=alert.model,
        day=alert.day,
        score=alert.score,
        predicted_label=alert.predicted_label,
        true_label=alert.true_label,
        techniques=alert.techniques or [],
    )
    if not alert.techniques:
        return AlertContext(alert=alert_out, matched_campaigns=[])

    edges = session.scalars(
        select(CampaignTechnique).where(CampaignTechnique.technique_id.in_(alert.techniques))
    ).all()
    matched: dict[str, list[str]] = {}
    for edge in edges:
        matched.setdefault(edge.campaign_id, []).append(edge.technique_id)

    campaigns = session.scalars(
        select(Campaign)
        .where(Campaign.campaign_id.in_(matched))
        .order_by(Campaign.report_count.desc())
    ).all()
    return AlertContext(
        alert=alert_out,
        matched_campaigns=[
            CampaignMatch(
                campaign_id=c.campaign_id,
                cve_ids=c.cve_ids,
                kev_cves=_kev_overlap(session, c.cve_ids),
                report_count=c.report_count,
                matched_techniques=sorted(matched[c.campaign_id]),
            )
            for c in campaigns
        ],
    )


class CampaignLinkOut(BaseModel):
    campaign_id: str
    matched_techniques: list[str]
    report_count: int
    kev_cves: list[str]


class HostThreatOut(BaseModel):
    host: str
    risk: int
    detectors: list[str]
    techniques: list[str]
    predicted_labels: list[str]
    true_labels: list[str]
    alert_count: int
    fused: list[CampaignLinkOut]
    simulated: bool


def _host_threat_out(threat: object) -> "HostThreatOut":
    from sentinel.correlate.hosts import HostThreat

    assert isinstance(threat, HostThreat)
    return HostThreatOut(
        host=threat.host,
        risk=threat.risk,
        detectors=threat.detectors,
        techniques=threat.techniques,
        predicted_labels=threat.predicted_labels,
        true_labels=threat.true_labels,
        alert_count=threat.alert_count,
        fused=[
            CampaignLinkOut(
                campaign_id=link.campaign_id,
                matched_techniques=link.matched_techniques,
                report_count=link.report_count,
                kev_cves=link.kev_cves,
            )
            for link in threat.fused
        ],
        simulated=threat.simulated,
    )


@app.get("/hosts")
def list_host_threats(session: SessionDep) -> list[HostThreatOut]:
    """Per-host threat rollup — alerts fused across detectors and against intel."""
    from sentinel.correlate.hosts import host_threats

    return [_host_threat_out(t) for t in host_threats(session, include_simulated=False)]


@app.get("/hosts/simulated")
def simulated_host_threats(session: SessionDep) -> list[HostThreatOut]:
    """Held-out host threats revealed on demand by the dashboard's simulate button."""
    from sentinel.correlate.hosts import host_threats

    everything = host_threats(session, include_simulated=True)
    return [_host_threat_out(t) for t in everything if t.simulated]


class TrendingOut(BaseModel):
    technique_id: str
    name: str | None
    recent_count: int
    prior_count: int
    lift: float


class DriftOut(BaseModel):
    population_stability_index: float
    verdict: str
    top_shifts: list[tuple[str, float]]


@app.get("/trending")
def trending(session: SessionDep, window_days: int = 7) -> list[TrendingOut]:
    from sentinel.correlate.trending import trending_techniques

    return [
        TrendingOut(
            technique_id=t.technique_id,
            name=t.name,
            recent_count=t.recent_count,
            prior_count=t.prior_count,
            lift=t.lift,
        )
        for t in trending_techniques(session, window_days=window_days)
    ]


@app.get("/feed-drift")
def feed_drift_endpoint(session: SessionDep, window_days: int = 7) -> DriftOut:
    from sentinel.correlate.trending import feed_drift

    drift = feed_drift(session, window_days=window_days)
    return DriftOut(
        population_stability_index=drift.population_stability_index,
        verdict=drift.verdict,
        top_shifts=drift.top_shifts,
    )


@app.get("/briefing", response_class=PlainTextResponse)
def briefing(session: SessionDep, window_days: int = 7) -> str:
    from sentinel.correlate.trending import briefing_lines, feed_drift, trending_techniques

    trending_list = trending_techniques(session, window_days=window_days)
    drift = feed_drift(session, window_days=window_days)
    campaigns = session.scalars(select(Campaign)).all()
    n_kev = sum(1 for c in campaigns if _kev_overlap(session, c.cve_ids))
    return "\n".join(briefing_lines(trending_list, drift, len(campaigns), n_kev))


@app.get("/attack-navigator-layer")
def attack_navigator_layer(session: SessionDep) -> dict[str, Any]:
    """ATT&CK Navigator layer JSON — import into MITRE's Navigator directly.

    Scores are evidence counts fused across tagged reports, campaign
    aggregation, and IDS alerts (same fusion the dashboard heatmap shows).
    """
    evidence: dict[str, int] = {}

    def bump(technique_id: str) -> None:
        evidence[technique_id] = evidence.get(technique_id, 0) + 1

    for technique_id in session.scalars(select(ReportTechnique.technique_id)):
        bump(technique_id)
    for technique_id in session.scalars(select(CampaignTechnique.technique_id)):
        bump(technique_id)
    for techniques in session.scalars(select(Alert.techniques)):
        for technique_id in techniques or []:
            bump(technique_id)

    return {
        "name": "SENTINEL technique evidence",
        "versions": {"attack": "16", "navigator": "5.1.0", "layer": "4.5"},
        "domain": "enterprise-attack",
        "description": (
            "Technique evidence fused across CTI reports, CVE-linked campaigns, and IDS alerts."
        ),
        "sorting": 3,
        "techniques": [
            {"techniqueID": technique_id, "score": count, "comment": f"evidence x{count}"}
            for technique_id, count in sorted(evidence.items())
        ],
        "gradient": {
            "colors": ["#ffffff", "#66b1ff", "#192fb3"],
            "minValue": 0,
            "maxValue": max(evidence.values(), default=1),
        },
    }


class TechniqueListItem(BaseModel):
    technique_id: str
    name: str
    tactics: list[str]


@app.get("/techniques")
def list_techniques(session: SessionDep) -> list[TechniqueListItem]:
    return [
        TechniqueListItem(technique_id=t.technique_id, name=t.name, tactics=t.tactics or [])
        for t in session.scalars(select(AttackTechnique))
    ]


@app.get("/techniques/{technique_id}")
def technique_detail(technique_id: str, session: SessionDep) -> TechniqueDetail:
    technique = session.get(AttackTechnique, technique_id)
    if technique is None:
        raise HTTPException(status_code=404, detail="technique not found")

    report_count = (
        session.scalar(
            select(func.count())
            .select_from(ReportTechnique)
            .where(ReportTechnique.technique_id == technique_id)
        )
        or 0
    )
    campaign_count = (
        session.scalar(
            select(func.count())
            .select_from(CampaignTechnique)
            .where(CampaignTechnique.technique_id == technique_id)
        )
        or 0
    )
    alert_count = 0
    for techniques in session.scalars(select(Alert.techniques)):
        if techniques and technique_id in techniques:
            alert_count += 1

    return TechniqueDetail(
        technique_id=technique.technique_id,
        name=technique.name,
        tactics=technique.tactics or [],
        url=technique.url,
        report_count=report_count,
        campaign_count=campaign_count,
        alert_count=alert_count,
    )
