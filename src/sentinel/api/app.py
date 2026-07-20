"""Read-only API over the SENTINEL knowledge graph.

Serves the fused threat picture: campaigns with corroborated techniques,
tagged reports, IDS alerts, and per-technique evidence across both layers.
Run locally: `make api` (http://localhost:8000/docs).
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from threading import Lock, Semaphore
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from sentinel.api.limits import RateLimiter
from sentinel.bridge.argus import OsintItem, osint_context
from sentinel.config import get_settings
from sentinel.correlate.fusion import (
    build_fusion_context,
    campaign_ages,
    score_campaign_matches,
)
from sentinel.db.base import get_session_factory
from sentinel.db.models import (
    Alert,
    AttackTechnique,
    CalibrationBatch,
    CalibrationFlow,
    CalibrationRun,
    Campaign,
    CampaignReport,
    CampaignTechnique,
    KevEntry,
    ReportTechnique,
    ThreatReport,
    Vulnerability,
)

_settings = get_settings()

# Reject a request body larger than this many bytes before it is buffered/parsed
# (UTF-8 worst case is 4 bytes/char, plus a little JSON overhead) so an oversized
# upload can't exhaust memory ahead of the precise per-field character check.
_MAX_BODY_BYTES = _settings.api_max_request_chars * 4 + 1024
_MAX_REQUEST_CHARS = _settings.api_max_request_chars

_rate_limiter = RateLimiter(
    _settings.api_rate_limit_requests, _settings.api_rate_limit_window_seconds
)
_calibration_rate_limiter = RateLimiter(
    _settings.api_calibration_rate_limit_requests,
    _settings.api_rate_limit_window_seconds,
)
_inference_sem = Semaphore(_settings.api_inference_concurrency)
_INFERENCE_TIMEOUT = _settings.api_inference_acquire_timeout_seconds


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Optionally warm the heavy mapper off the request path so the first public
    # request is fast. Best-effort and non-blocking: a failure here (e.g. no
    # model cache) just leaves the lazy load to the first request.
    if _settings.api_warm_model:
        from threading import Thread

        def _warm() -> None:
            session = get_session_factory()()
            try:
                _get_mapper(session)
            except Exception:  # warm-up is best-effort; first request retries
                pass
            finally:
                session.close()

        Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    title="SENTINEL",
    description="Cyber threat intelligence fusion: OSINT + NLP + IDS in one ATT&CK graph",
    version="0.1.0",
    lifespan=lifespan,
)

# In development allow any localhost port (the Vite dev-server port varies). In
# production set SENTINEL_API_ALLOWED_ORIGINS to the deployed dashboard's exact
# origin(s) — otherwise the browser blocks every cross-origin call.
if _settings.api_allowed_origins.strip():
    _origins = [o.strip() for o in _settings.api_allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def limit_body_size(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > _MAX_BODY_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)
    return await call_next(request)


def get_session() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_session)]
# Analytics lookback window, bounded so a query param can't request a degenerate
# (<=0) or absurd range.
WindowDays = Annotated[int, Query(ge=1, le=365)]


def _client_key(request: Request) -> str:
    # Behind a *trusted* reverse proxy the real client is the first
    # X-Forwarded-For hop. Only honour it when api_trust_forwarded_header is set,
    # otherwise the socket peer is used: on a directly-exposed server the header
    # is attacker-controlled and rotating it would defeat the rate limit.
    if _settings.api_trust_forwarded_header:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# The technique mapper loads SecureBERT (~GB) and encodes 697 ATT&CK docs once.
# Lazy-build it on first /map-techniques call and cache the singleton for reuse;
# the lock guards against a double-build under FastAPI's threadpool.
_mapper: Any = None
_mapper_lock = Lock()


def _get_mapper(session: Session) -> Any:
    global _mapper
    if _mapper is None:
        with _mapper_lock:
            if _mapper is None:
                from sentinel.nlp.encoders import BiEncoder
                from sentinel.nlp.mapper import TechniqueMapper, load_technique_docs

                _mapper = TechniqueMapper(
                    load_technique_docs(session),
                    encoder=BiEncoder(),
                    cache_dir=_settings.nlp_embedding_cache_dir,
                    model_name=_settings.nlp_bi_encoder_model,
                    lexical=True,  # hybrid retrieval, matching the tagging pipeline
                )
    return _mapper


def _get_calibration_pack() -> Any:
    """Lazy-load the optional ML pack without pulling ML deps into the slim API."""
    from sentinel.ids.calibrate import load_pack

    return load_pack(_settings.calibration_pack_path)


def _guard_calibration(request: Request) -> None:
    if not _settings.api_enable_calibration:
        raise HTTPException(status_code=404, detail="calibration workflow is disabled")
    if not _calibration_rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="calibration rate limit exceeded")


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
    # Age in days of the campaign's most recent member report (None if undated);
    # lets the dashboard rank campaigns by how active they are right now.
    age_days: float | None


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


class FusionScoreOut(BaseModel):
    """Explainable fusion strength: the headline [0,1] confidence and its parts."""

    strength: float
    specificity: float
    recency: float
    corroboration: float
    age_days: float | None


class CampaignMatch(BaseModel):
    campaign_id: str
    cve_ids: list[str]
    kev_cves: list[str]
    report_count: int
    matched_techniques: list[str]
    fusion: FusionScoreOut


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
def health(session: SessionDep) -> dict[str, str]:
    """Readiness check: confirms the process is up *and* the database answers, so
    a load balancer can tell "serving" from merely "running"."""
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:  # DB down/unreachable — report unready, don't 500
        raise HTTPException(status_code=503, detail="database unavailable") from exc
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
    """Campaigns ranked the way an analyst triages them: actively-exploited (KEV)
    first, then most recently reported, then most corroborated."""
    ctx = build_fusion_context(session)
    summaries = [
        CampaignSummary(
            campaign_id=c.campaign_id,
            cve_ids=c.cve_ids,
            kev_cves=ctx.kev_by_campaign.get(c.campaign_id, []),
            report_count=c.report_count,
            techniques=_campaign_techniques(session, c.campaign_id),
            age_days=ctx.ages.get(c.campaign_id),
        )
        for c in session.scalars(select(Campaign))
    ]
    summaries.sort(
        key=lambda s: (
            -len(s.kev_cves),  # most actively-exploited CVEs first
            s.age_days if s.age_days is not None else float("inf"),  # then freshest
            -s.report_count,  # then best-corroborated
        )
    )
    return summaries


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
        # Same corpus-anchored age the /campaigns list uses, via one query
        # rather than building the whole fusion context for a single campaign.
        age_days=campaign_ages(session).get(campaign_id),
        reports=[_report_summary(session, r) for r in reports],
    )


@app.get("/campaigns/{campaign_id}/osint")
def campaign_osint(
    campaign_id: str,
    request: Request,
    session: SessionDep,
    limit: int = Query(default=5, ge=1, le=20),
) -> list[OsintItem]:
    """Open-source context for a cyber campaign, fused read-only from the sibling ARGUS
    workbench — the reverse of ARGUS pulling cyber evidence, closing the all-source loop.
    The query is built from the campaign's report titles (what it is about); ARGUS returns
    source-rated OSINT relevant to it. Empty when the bridge is off (``SENTINEL_ARGUS_API_URL``
    unset) or ARGUS is unreachable — never breaks the route.

    Rate-limited like /map-techniques: each hit makes a blocking outbound call to
    ARGUS, so an unthrottled public client could hold worker threads and turn
    SENTINEL into a traffic amplifier against the sibling service."""
    if not _rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")
    campaign = session.get(Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    member_ids = session.scalars(
        select(CampaignReport.report_id).where(CampaignReport.campaign_id == campaign_id)
    ).all()
    titles = session.scalars(
        select(ThreatReport.title).where(ThreatReport.report_id.in_(member_ids))
    ).all()
    query = " ".join(t for t in titles if t) or " ".join(campaign.cve_ids)
    return osint_context(query, limit=limit)


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
        .limit(min(max(limit, 1), 200))  # floor at 1: a negative LIMIT errors on Postgres
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
    query = (
        select(Alert)
        .order_by(Alert.score.desc())
        .limit(min(max(limit, 1), 200))  # floor at 1: a negative LIMIT errors on Postgres
        .offset(max(offset, 0))
    )
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
    context. Matches are not raw tag overlap: each is scored by the rarity of the
    shared technique(s), the campaign's freshness, and how firmly the campaign
    asserts them, and ranked by the combined fusion strength so a specific, recent,
    well-corroborated correlation surfaces above a coincidental shared tag.
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
    matches = score_campaign_matches(session, set(alert.techniques or []))
    return AlertContext(
        alert=alert_out,
        matched_campaigns=[
            CampaignMatch(
                campaign_id=m.campaign_id,
                cve_ids=m.cve_ids,
                kev_cves=m.kev_cves,
                report_count=m.report_count,
                matched_techniques=m.matched_techniques,
                fusion=FusionScoreOut(
                    strength=m.fusion.strength,
                    specificity=m.fusion.specificity,
                    recency=m.fusion.recency,
                    corroboration=m.fusion.corroboration,
                    age_days=m.fusion.age_days,
                ),
            )
            for m in matches
        ],
    )


class CampaignLinkOut(BaseModel):
    campaign_id: str
    matched_techniques: list[str]
    report_count: int
    kev_cves: list[str]
    fusion: FusionScoreOut


class AlertRefOut(BaseModel):
    alert_id: int
    model: str
    score: float
    predicted_label: str | None
    techniques: list[str]


class HostThreatOut(BaseModel):
    host: str
    risk: int
    detectors: list[str]
    techniques: list[str]
    predicted_labels: list[str]
    true_labels: list[str]
    alert_count: int
    # Strongest detection per detector — each drillable via /alerts/{id}/context.
    alerts: list[AlertRefOut]
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
        alerts=[
            AlertRefOut(
                alert_id=a.alert_id,
                model=a.model,
                score=a.score,
                predicted_label=a.predicted_label,
                techniques=a.techniques,
            )
            for a in threat.alerts
        ],
        fused=[
            CampaignLinkOut(
                campaign_id=link.campaign_id,
                matched_techniques=link.matched_techniques,
                report_count=link.report_count,
                kev_cves=link.kev_cves,
                fusion=FusionScoreOut(
                    strength=link.fusion.strength,
                    specificity=link.fusion.specificity,
                    recency=link.fusion.recency,
                    corroboration=link.fusion.corroboration,
                    age_days=link.fusion.age_days,
                ),
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
def trending(session: SessionDep, window_days: WindowDays = 7) -> list[TrendingOut]:
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
def feed_drift_endpoint(session: SessionDep, window_days: WindowDays = 7) -> DriftOut:
    from sentinel.correlate.trending import feed_drift

    drift = feed_drift(session, window_days=window_days)
    return DriftOut(
        population_stability_index=drift.population_stability_index,
        verdict=drift.verdict,
        top_shifts=drift.top_shifts,
    )


@app.get("/briefing", response_class=PlainTextResponse)
def briefing(session: SessionDep, window_days: WindowDays = 7) -> str:
    from sentinel.correlate.trending import briefing_lines, feed_drift, trending_techniques

    trending_list = trending_techniques(session, window_days=window_days)
    drift = feed_drift(session, window_days=window_days)
    campaigns = session.scalars(select(Campaign)).all()
    # One batched KEV lookup instead of a query per campaign.
    all_cves = {cve for c in campaigns for cve in c.cve_ids}
    kev_listed = (
        set(session.scalars(select(KevEntry.cve_id).where(KevEntry.cve_id.in_(all_cves))))
        if all_cves
        else set()
    )
    n_kev = sum(1 for c in campaigns if any(cve in kev_listed for cve in c.cve_ids))
    return "\n".join(briefing_lines(trending_list, drift, len(campaigns), n_kev))


@app.get("/attack-navigator-layer")
def attack_navigator_layer(session: SessionDep) -> dict[str, Any]:
    """ATT&CK Navigator layer JSON — import into MITRE's Navigator directly.

    Scores are evidence counts fused across tagged reports, campaign
    aggregation, and IDS alerts (same fusion the dashboard heatmap shows).
    """
    evidence: dict[str, int] = {}

    def add(technique_id: str, n: int) -> None:
        evidence[technique_id] = evidence.get(technique_id, 0) + n

    # Report + campaign edges aggregate directly in SQL (indexed technique_id).
    for tid, n in session.execute(
        select(ReportTechnique.technique_id, func.count()).group_by(ReportTechnique.technique_id)
    ):
        add(tid, n)
    for tid, n in session.execute(
        select(CampaignTechnique.technique_id, func.count()).group_by(
            CampaignTechnique.technique_id
        )
    ):
        add(tid, n)
    # Alert techniques are a JSON array. On Postgres, unnest + group in SQL;
    # elsewhere (SQLite in tests) scan the arrays in Python.
    if session.get_bind().dialect.name == "postgresql":
        rows = session.execute(
            text(
                "SELECT elem, count(*) FROM alerts, "
                "jsonb_array_elements_text(techniques) AS elem GROUP BY elem"
            )
        )
        for tid, n in rows:
            add(tid, n)
    else:
        for techniques in session.scalars(select(Alert.techniques)):
            for technique_id in techniques or []:
                add(technique_id, 1)

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


def _alert_count_for_technique(session: Session, technique_id: str) -> int:
    """Number of alerts whose technique list contains `technique_id`.

    On Postgres this is a JSONB containment (`@>`) query backed by the GIN index
    on `alerts.techniques` (migration 0009) — O(matches). On other dialects
    (SQLite in tests) it falls back to scanning the JSON arrays in Python.
    """
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy import cast
        from sqlalchemy.dialects.postgresql import JSONB

        stmt = (
            select(func.count())
            .select_from(Alert)
            .where(Alert.techniques.op("@>")(cast([technique_id], JSONB)))
        )
        return session.scalar(stmt) or 0
    return sum(
        1
        for techniques in session.scalars(select(Alert.techniques))
        if techniques and technique_id in techniques
    )


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
    alert_count = _alert_count_for_technique(session, technique_id)

    return TechniqueDetail(
        technique_id=technique.technique_id,
        name=technique.name,
        tactics=technique.tactics or [],
        url=technique.url,
        report_count=report_count,
        campaign_count=campaign_count,
        alert_count=alert_count,
    )


class MapRequest(BaseModel):
    text: str = Field(max_length=_MAX_REQUEST_CHARS)


class MappedTechnique(BaseModel):
    technique_id: str
    name: str
    score: float
    corroborations: int
    tactics: list[str]
    url: str | None


CalibrationStrategy = Literal[
    "random", "random-blind", "active", "coreset", "cluster", "stratified"
]
CalibrationLabel = Literal["benign", "attack"]


class CalibrationBatchCreate(BaseModel):
    n: int = Field(default=50, ge=1, le=200)
    strategy: CalibrationStrategy = "stratified"
    # np.random.default_rng rejects negative seeds; validate here so a bad seed
    # is a 422 instead of masquerading as a 503 pack failure.
    seed: int = Field(default=13, ge=0)
    notes: str | None = Field(default=None, max_length=500)


class CalibrationLabelRequest(BaseModel):
    label: CalibrationLabel


class CalibrationFlowResponse(BaseModel):
    id: int
    pool_row: int
    features: dict[str, float]
    model_score: float
    operator_label: str | None
    true_label: str | None = None
    labelled_at: datetime | None


class CalibrationRunResponse(BaseModel):
    id: int
    created_at: datetime
    recall_before: float
    recall_after: float
    fpr_after: float
    auc_after: float
    n_labels_used: int
    operator_accuracy: float
    metrics: dict[str, Any]


class CalibrationBatchResponse(BaseModel):
    id: int
    created_at: datetime
    strategy: str
    seed: int
    n_flows: int
    n_labelled: int
    status: str
    notes: str | None
    flows: list[CalibrationFlowResponse]
    runs: list[CalibrationRunResponse]


class CalibrationCurveResponse(BaseModel):
    strategy: str
    points: list[dict[str, Any]]


def _flow_response(flow: CalibrationFlow) -> CalibrationFlowResponse:
    # Truth is withheld for unseen rows. Once the operator commits a label it
    # is safe to reveal for immediate training feedback and later review.
    revealed = flow.true_label if flow.operator_label is not None else None
    return CalibrationFlowResponse(
        id=flow.id,
        pool_row=flow.pool_row,
        features=flow.features,
        model_score=flow.model_score,
        operator_label=flow.operator_label,
        true_label=revealed,
        labelled_at=flow.labelled_at,
    )


def _run_response(run: CalibrationRun) -> CalibrationRunResponse:
    return CalibrationRunResponse(
        id=run.id,
        created_at=run.created_at,
        recall_before=run.recall_before,
        recall_after=run.recall_after,
        fpr_after=run.fpr_after,
        auc_after=run.auc_after,
        n_labels_used=run.n_labels_used,
        operator_accuracy=run.operator_accuracy,
        metrics=run.metrics,
    )


def _batch_response(session: Session, batch: CalibrationBatch) -> CalibrationBatchResponse:
    flows = list(
        session.scalars(
            select(CalibrationFlow)
            .where(CalibrationFlow.batch_id == batch.id)
            .order_by(CalibrationFlow.id)
        )
    )
    runs = list(
        session.scalars(
            select(CalibrationRun)
            .where(CalibrationRun.batch_id == batch.id)
            .order_by(CalibrationRun.id.desc())
        )
    )
    return CalibrationBatchResponse(
        id=batch.id,
        created_at=batch.created_at,
        strategy=batch.strategy,
        seed=batch.seed,
        n_flows=batch.n_flows,
        n_labelled=sum(flow.operator_label is not None for flow in flows),
        status=batch.status,
        notes=batch.notes,
        flows=[_flow_response(flow) for flow in flows],
        runs=[_run_response(run) for run in runs],
    )


def _label_flow(
    flow_id: int,
    label: CalibrationLabel,
    session: Session,
) -> CalibrationFlowResponse:
    flow = session.get(CalibrationFlow, flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail="calibration flow not found")
    flow.operator_label = label
    flow.labelled_at = datetime.now().astimezone()
    labelled = session.scalar(
        select(func.count())
        .select_from(CalibrationFlow)
        .where(
            CalibrationFlow.batch_id == flow.batch_id,
            CalibrationFlow.operator_label.is_not(None),
        )
    )
    batch = session.get(CalibrationBatch, flow.batch_id)
    # Promote open -> labelled once every flow has an answer; never demote a
    # batch that has already been retrained.
    if batch is not None and batch.status == "open" and (labelled or 0) >= batch.n_flows:
        batch.status = "labelled"
    session.commit()
    session.refresh(flow)
    return _flow_response(flow)


@app.post(
    "/calibration/batches",
    response_model=CalibrationBatchResponse,
    response_model_exclude_none=True,
)
def create_calibration_batch(
    req: CalibrationBatchCreate,
    request: Request,
    session: SessionDep,
) -> CalibrationBatchResponse:
    """Sample a blind, reproducible flow-labelling batch from the frozen pack."""
    _guard_calibration(request)
    try:
        from sentinel.ids.calibrate import sample_batch

        pack = _get_calibration_pack()
        rows = sample_batch(pack, n=req.n, strategy=req.strategy, seed=req.seed)
    except (FileNotFoundError, ImportError, KeyError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="calibration pack unavailable") from exc

    batch = CalibrationBatch(
        strategy=req.strategy,
        seed=req.seed,
        n_flows=len(rows),
        status="open",
        notes=req.notes,
    )
    session.add(batch)
    session.flush()
    for row_value in rows:
        row = int(row_value)
        features = {name: float(pack.pool_x.iloc[row][name]) for name in pack.feature_names}
        session.add(
            CalibrationFlow(
                batch_id=batch.id,
                pool_row=row,
                features=features,
                model_score=float(pack.pool_scores[row]),
                true_label="attack" if int(pack.pool_y[row]) == 1 else "benign",
            )
        )
    session.commit()
    session.refresh(batch)
    return _batch_response(session, batch)


@app.get(
    "/calibration/batches/{batch_id}",
    response_model=CalibrationBatchResponse,
    response_model_exclude_none=True,
)
def get_calibration_batch(
    batch_id: int, request: Request, session: SessionDep
) -> CalibrationBatchResponse:
    _guard_calibration(request)
    batch = session.get(CalibrationBatch, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="calibration batch not found")
    return _batch_response(session, batch)


@app.post(
    "/calibration/flows/{flow_id}/label",
    response_model=CalibrationFlowResponse,
    response_model_exclude_none=True,
)
def label_calibration_flow(
    flow_id: int,
    req: CalibrationLabelRequest,
    request: Request,
    session: SessionDep,
) -> CalibrationFlowResponse:
    """Commit or replace one operator label, then reveal that row's truth."""
    _guard_calibration(request)
    return _label_flow(flow_id, req.label, session)


@app.post(
    "/calibration/flows/{flow_id}/simulate-label",
    response_model=CalibrationFlowResponse,
    response_model_exclude_none=True,
)
def simulate_calibration_label(
    flow_id: int, request: Request, session: SessionDep
) -> CalibrationFlowResponse:
    """Demo-only shortcut: apply the pack's hidden truth as the operator label."""
    _guard_calibration(request)
    flow = session.get(CalibrationFlow, flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail="calibration flow not found")
    label: CalibrationLabel = "attack" if flow.true_label == "attack" else "benign"
    return _label_flow(flow_id, label, session)


@app.post(
    "/calibration/batches/{batch_id}/retrain",
    response_model=CalibrationRunResponse,
)
def retrain_calibration_batch(
    batch_id: int, request: Request, session: SessionDep
) -> CalibrationRunResponse:
    """Fit source plus operator labels and grade once on the held-out target test."""
    _guard_calibration(request)
    batch = session.get(CalibrationBatch, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="calibration batch not found")
    flows = list(
        session.scalars(
            select(CalibrationFlow).where(
                CalibrationFlow.batch_id == batch.id,
                CalibrationFlow.operator_label.is_not(None),
            )
        )
    )
    if not flows:
        raise HTTPException(status_code=422, detail="label at least one flow before retraining")

    if not _inference_sem.acquire(timeout=_INFERENCE_TIMEOUT):
        raise HTTPException(status_code=503, detail="calibration worker busy, try again shortly")
    try:
        from sentinel.ids.calibrate import retrain

        pack = _get_calibration_pack()
        labelled = [(flow.pool_row, 1 if flow.operator_label == "attack" else 0) for flow in flows]
        metrics = retrain(pack, labelled, seed=batch.seed)
    except (FileNotFoundError, ImportError, KeyError, OSError) as exc:
        raise HTTPException(status_code=503, detail="calibration worker unavailable") from exc
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        _inference_sem.release()

    run = CalibrationRun(
        batch_id=batch.id,
        recall_before=metrics.recall_before,
        recall_after=metrics.recall_after,
        fpr_after=metrics.fpr_after,
        auc_after=metrics.auc_after,
        n_labels_used=metrics.n_labels_used,
        operator_accuracy=metrics.operator_accuracy,
        metrics=metrics.details(),
    )
    batch.status = "retrained"
    session.add(run)
    session.commit()
    session.refresh(run)
    return _run_response(run)


@app.get("/calibration/curve", response_model=CalibrationCurveResponse)
def calibration_curve(request: Request) -> CalibrationCurveResponse:
    """Return the recorded multi-seed WS3 reference curve for context."""
    _guard_calibration(request)
    try:
        from sentinel.ids.calibrate import LABEL_EFFICIENCY_CURVE
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="calibration workflow unavailable") from exc
    return CalibrationCurveResponse(strategy="stratified", points=LABEL_EFFICIENCY_CURVE)


@app.post("/map-techniques")
def map_techniques(req: MapRequest, request: Request, session: SessionDep) -> list[MappedTechnique]:
    """Run the zero-shot ATT&CK mapper over pasted CTI text.

    Inspects the supplied text only — it does not fetch or scan any URL. Splits
    into sentences, maps each through the SecureBERT + BM25 hybrid retriever, and
    corroborates the evidence across sentences. Returns the top techniques with
    their score so the UI can show the same mapper that tags ingested reports.

    Hardened for public exposure: per-client rate limit, a bounded concurrency
    cap on the model so load sheds as 503 instead of exhausting memory, and a
    503 (not a 500 stack trace) if the model can't be loaded.
    """
    if not _rate_limiter.allow(_client_key(request)):
        raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")

    # The NLP stack (numpy/sentence-transformers/…) is absent from the slim deploy
    # image on purpose, so importing the mapper fails there — degrade to 503 rather
    # than a 500, same as a model-load failure below.
    try:
        from sentinel.nlp.mapper import aggregate_matches
        from sentinel.nlp.tagging import split_sentences
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="technique mapper unavailable") from exc

    sentences = split_sentences(req.text)
    if not sentences:
        return []

    # Bound concurrent model runs: shed load gracefully rather than letting many
    # simultaneous inferences pile up and run the box out of memory.
    if not _inference_sem.acquire(timeout=_INFERENCE_TIMEOUT):
        raise HTTPException(status_code=503, detail="mapper busy, try again shortly")
    try:
        mapper = _get_mapper(session)
        aggregated = aggregate_matches(mapper.map_text(s, top_k=5) for s in sentences)[:8]
    except Exception as exc:  # model load / inference failure — never leak a 500
        raise HTTPException(status_code=503, detail="technique mapper unavailable") from exc
    finally:
        _inference_sem.release()

    if not aggregated:
        return []

    meta = {
        t.technique_id: t
        for t in session.scalars(
            select(AttackTechnique).where(
                AttackTechnique.technique_id.in_([m.technique_id for m in aggregated])
            )
        )
    }
    results = []
    for m in aggregated:
        t = meta.get(m.technique_id)
        results.append(
            MappedTechnique(
                technique_id=m.technique_id,
                name=m.name,
                score=m.score,
                corroborations=m.corroborations,
                tactics=(t.tactics or []) if t else [],
                url=t.url if t else None,
            )
        )
    return results


# Serve the built React dashboard from the API's own origin when a dist dir is
# configured (the single-service cloud image). Mounted last so every API route
# above takes precedence; the SPA and its assets are served for everything else,
# so the dashboard and API share a host and need no CORS or second service.
# Empty in dev/tests, where the Vite dev server runs separately.
if _settings.api_dashboard_dist:
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_settings.api_dashboard_dist, html=True), name="dashboard")
