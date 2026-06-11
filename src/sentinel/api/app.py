"""Read-only API over the SENTINEL knowledge graph.

Serves the fused threat picture: campaigns with corroborated techniques,
tagged reports, IDS alerts, and per-technique evidence across both layers.
Run locally: `make api` (http://localhost:8000/docs).
"""

from collections.abc import Iterator
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
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


class TechniqueEvidence(BaseModel):
    technique_id: str
    name: str | None
    score: float
    corroborations: int


class CampaignSummary(BaseModel):
    campaign_id: str
    cve_ids: list[str]
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
    )


@app.get("/campaigns")
def list_campaigns(session: SessionDep) -> list[CampaignSummary]:
    campaigns = session.scalars(select(Campaign).order_by(Campaign.report_count.desc())).all()
    return [
        CampaignSummary(
            campaign_id=c.campaign_id,
            cve_ids=c.cve_ids,
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
        report_count=campaign.report_count,
        techniques=_campaign_techniques(session, campaign_id),
        reports=[_report_summary(session, r) for r in reports],
    )


@app.get("/reports")
def list_reports(
    session: SessionDep,
    source: str | None = None,
    limit: int = 50,
) -> list[ReportSummary]:
    query = select(ThreatReport).order_by(ThreatReport.ingested_at.desc()).limit(min(limit, 200))
    if source is not None:
        query = query.where(ThreatReport.source == source)
    return [_report_summary(session, r) for r in session.scalars(query).all()]


@app.get("/alerts")
def list_alerts(
    session: SessionDep,
    model: str | None = None,
    limit: int = 50,
) -> list[AlertOut]:
    query = select(Alert).order_by(Alert.score.desc()).limit(min(limit, 200))
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
