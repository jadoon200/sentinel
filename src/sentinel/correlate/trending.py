"""Temporal analytics over the knowledge graph: technique drift and trending.

Two questions an analyst asks of an intel feed that a static graph can't
answer: *what's surging right now* (techniques whose mention rate is rising)
and *is the feed itself drifting* (are we ingesting a different threat mix
than last week — model-monitoring for the CTI side, mirroring the IDS
benign-drift detector). Both are computed from report publish/ingest times
with no extra storage.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.db.models import AttackTechnique, ReportTechnique, ThreatReport


@dataclass(frozen=True)
class TrendingTechnique:
    technique_id: str
    name: str | None
    recent_count: int
    prior_count: int
    lift: float  # recent rate / prior rate, smoothed


def _naive(ts: datetime) -> datetime:
    # SQLite drops tzinfo on round-trip while Postgres keeps it; normalize so
    # window comparisons never mix naive and aware datetimes.
    return ts.replace(tzinfo=None) if ts.tzinfo is not None else ts


def _report_times(session: Session) -> dict[str, datetime]:
    rows = session.execute(select(ThreatReport.report_id, ThreatReport.ingested_at))
    return {report_id: _naive(ts) for report_id, ts in rows}


def trending_techniques(
    session: Session,
    now: datetime | None = None,
    window_days: int = 7,
    top_k: int = 15,
) -> list[TrendingTechnique]:
    """Techniques whose mention rate in the last `window_days` exceeds the prior window.

    Lift = (recent + 1) / (prior + 1), so a technique appearing 5x this week and
    0x before scores 6.0; one steady at 5/5 scores ~1.0. The +1 smoothing keeps
    a single new mention from dominating.
    """
    times = _report_times(session)
    now = _naive(now) if now else max(times.values(), default=datetime.now())
    recent_cut = now - timedelta(days=window_days)
    prior_cut = now - timedelta(days=2 * window_days)

    recent: dict[str, int] = {}
    prior: dict[str, int] = {}
    edges = session.execute(select(ReportTechnique.report_id, ReportTechnique.technique_id))
    for report_id, technique_id in edges:
        ts = times.get(report_id)
        if ts is None:
            continue
        if ts >= recent_cut:
            recent[technique_id] = recent.get(technique_id, 0) + 1
        elif ts >= prior_cut:
            prior[technique_id] = prior.get(technique_id, 0) + 1

    names = {
        tid: name
        for tid, name in session.execute(select(AttackTechnique.technique_id, AttackTechnique.name))
    }
    trending = [
        TrendingTechnique(
            technique_id=technique_id,
            name=names.get(technique_id),
            recent_count=recent.get(technique_id, 0),
            prior_count=prior.get(technique_id, 0),
            lift=(recent.get(technique_id, 0) + 1) / (prior.get(technique_id, 0) + 1),
        )
        for technique_id in set(recent) | set(prior)
    ]
    trending.sort(key=lambda t: (t.lift, t.recent_count), reverse=True)
    return trending[:top_k]


@dataclass(frozen=True)
class DriftReport:
    population_stability_index: float
    verdict: str  # "stable" | "moderate" | "significant"
    top_shifts: list[tuple[str, float]]  # (source, recent_share - prior_share)


def _psi(recent: dict[str, float], prior: dict[str, float]) -> float:
    """Population Stability Index over a categorical distribution (sources)."""
    keys = set(recent) | set(prior)
    total = 0.0
    for key in keys:
        r = max(recent.get(key, 0.0), 1e-6)
        p = max(prior.get(key, 0.0), 1e-6)
        total += (r - p) * math.log(r / p)
    return total


def feed_drift(
    session: Session,
    now: datetime | None = None,
    window_days: int = 7,
) -> DriftReport:
    """PSI of the report-source mix, recent window vs the prior window.

    PSI is the standard model-monitoring drift score: < 0.1 stable, 0.1-0.25
    moderate, > 0.25 significant. Applied here to the *source* distribution,
    it flags when the feed composition shifts (a source goes quiet, a new one
    floods) — the CTI analogue of the IDS benign-traffic drift the conformal
    controller handles.
    """
    times = _report_times(session)
    now = _naive(now) if now else max(times.values(), default=datetime.now())
    recent_cut = now - timedelta(days=window_days)
    prior_cut = now - timedelta(days=2 * window_days)

    rows = session.execute(select(ThreatReport.report_id, ThreatReport.source))
    recent_counts: dict[str, float] = {}
    prior_counts: dict[str, float] = {}
    for report_id, source in rows:
        ts = times.get(report_id)
        if ts is None:
            continue
        if ts >= recent_cut:
            recent_counts[source] = recent_counts.get(source, 0.0) + 1
        elif ts >= prior_cut:
            prior_counts[source] = prior_counts.get(source, 0.0) + 1

    recent_share = _normalize(recent_counts)
    prior_share = _normalize(prior_counts)
    psi = _psi(recent_share, prior_share)
    verdict = "stable" if psi < 0.1 else "moderate" if psi < 0.25 else "significant"
    shifts = sorted(
        (
            (s, recent_share.get(s, 0.0) - prior_share.get(s, 0.0))
            for s in recent_share | prior_share
        ),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    return DriftReport(population_stability_index=psi, verdict=verdict, top_shifts=shifts[:5])


def _normalize(counts: dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def briefing_lines(
    trending: Sequence[TrendingTechnique], drift: DriftReport, n_campaigns: int, n_kev: int
) -> list[str]:
    """Plain-text intelligence briefing from the analytics — the SOC handoff."""
    lines = ["# SENTINEL daily threat briefing", ""]
    lines.append(
        f"- Feed status: **{drift.verdict}** "
        f"(source-mix PSI {drift.population_stability_index:.3f})"
    )
    lines.append(f"- {n_campaigns} active campaigns; {n_kev} involve actively-exploited (KEV) CVEs")
    if trending:
        lines.append("")
        lines.append("## Trending techniques (last window vs prior)")
        for t in trending[:5]:
            name = t.name or t.technique_id
            lines.append(
                f"- **{t.technique_id} {name}** — {t.recent_count} mentions "
                f"(was {t.prior_count}, lift x{t.lift:.1f})"
            )
    if drift.verdict != "stable" and drift.top_shifts:
        lines.append("")
        lines.append("## Feed composition shift")
        for source, delta in drift.top_shifts[:3]:
            arrow = "up" if delta > 0 else "down"
            lines.append(f"- `{source}` {arrow} {abs(delta) * 100:.0f} pts of feed share")
    return lines
