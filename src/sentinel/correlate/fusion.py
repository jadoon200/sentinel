"""Deep fusion scoring: turn a shared ATT&CK tag into a calibrated correlation.

The naive fusion is set overlap — an alert and a campaign "match" if they share
any technique. That makes T1110 (brute force, present in nearly every campaign)
count exactly as much as T1195.001 (supply-chain compromise, rare and specific),
so a reviewer is right to ask "lots of campaigns involve DoS — why is this match
meaningful?". This module answers that by weighting a match on three independent,
interpretable signals and combining them into a [0, 1] fusion strength:

  specificity   — IDF rarity of the shared technique(s) over the report corpus,
                  so a *surprising* shared tag counts more than a ubiquitous one
  recency       — exponential decay on the matched campaign's age, so a *live*
                  correlation outranks a months-old one
  corroboration — how strongly the campaign itself asserts the shared technique
                  (member-report count and mean technique score)

    strength = (specificity * recency * corroboration) ** (1/3)

The geometric mean is conjunctive by construction — a near-zero factor drags the
whole score down — so a strong correlation must be rare AND recent AND
well-evidenced, not merely share a common tag. Every component is returned
alongside the headline number so the dashboard can explain *why* a match scored
the way it did rather than presenting an opaque confidence.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.config import get_settings
from sentinel.db.models import (
    Campaign,
    CampaignReport,
    CampaignTechnique,
    KevEntry,
    ReportTechnique,
    ThreatReport,
)


@dataclass(frozen=True)
class FusionScore:
    """The fusion strength and its three explainable components, each in [0, 1]."""

    strength: float
    specificity: float
    recency: float
    corroboration: float
    age_days: float | None  # campaign age the recency factor was derived from


@dataclass(frozen=True)
class ScoredMatch:
    campaign_id: str
    cve_ids: list[str]
    kev_cves: list[str]
    report_count: int
    matched_techniques: list[str]
    fusion: FusionScore = field(default_factory=lambda: FusionScore(0.0, 0.0, 0.0, 0.0, None))


def _naive(ts: datetime) -> datetime:
    # SQLite drops tzinfo on round-trip while Postgres keeps it; normalize so
    # age arithmetic never mixes naive and aware datetimes.
    return ts.replace(tzinfo=None) if ts.tzinfo is not None else ts


def _soft_or(values: list[float]) -> float:
    """Probabilistic OR: 1 - prod(1 - v). Multiple weak signals compound, but the
    result stays in [0, 1] and is dominated by the strongest evidence."""
    product = 1.0
    for value in values:
        product *= 1.0 - min(max(value, 0.0), 1.0)
    return 1.0 - product


def _corroboration_factor(score: float, corroborations: int) -> float:
    """How firmly the campaign asserts a technique: its mean technique score
    discounted by a saturating function of the member-report count, so one report
    counts for less than three but extra corroboration has diminishing returns."""
    return min(max(score, 0.0), 1.0) * (1.0 - 0.5 ** max(corroborations, 0))


def _recency_factor(age_days: float | None, half_life_days: float) -> float:
    """Exponential decay: a freshly reported campaign scores ~1.0, one a half-life
    old scores 0.5. Unknown age (no datable member reports) is treated as neutral
    rather than penalized, since absence of a timestamp is not staleness."""
    if age_days is None:
        return 1.0
    if age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


def technique_idf(session: Session) -> dict[str, float]:
    """Normalized inverse-document-frequency of each technique over the report corpus.

    A technique mentioned in few reports is "surprising" (idf -> 1.0); one in most
    reports is generic (idf -> 0.0). Document frequency is the number of distinct
    reports tagged with the technique (report_techniques has a (report, technique)
    primary key, so each edge is one distinct report). Values are min-max scaled
    across the observed vocabulary so specificity reads as "rare relative to this
    feed" rather than an unbounded log.
    """
    edges = session.execute(select(ReportTechnique.report_id, ReportTechnique.technique_id)).all()
    if not edges:
        return {}

    n_reports = len({report_id for report_id, _ in edges})
    doc_freq: dict[str, int] = {}
    for _, technique_id in edges:
        doc_freq[technique_id] = doc_freq.get(technique_id, 0) + 1

    raw = {tid: math.log((n_reports + 1) / (df + 1)) + 1.0 for tid, df in doc_freq.items()}
    lo, hi = min(raw.values()), max(raw.values())
    if hi == lo:  # every technique equally common — no discriminating rarity
        return {tid: 1.0 for tid in raw}
    return {tid: (value - lo) / (hi - lo) for tid, value in raw.items()}


def _campaign_ages(
    session: Session, campaign_ids: set[str], now: datetime | None
) -> tuple[dict[str, float], datetime | None]:
    """Age in days of each campaign's most recent member report.

    A campaign's "freshness" is the latest publish (falling back to ingest) time
    across its reports. When `now` is not supplied it defaults to the newest
    observation in the matched set, so scoring is deterministic and dataset-anchored
    (the same convention the trending analytics use) rather than wall-clock drifting.
    """
    if not campaign_ids:
        return {}, now
    rows = session.execute(
        select(
            CampaignReport.campaign_id,
            ThreatReport.published,
            ThreatReport.ingested_at,
        )
        .join(ThreatReport, ThreatReport.report_id == CampaignReport.report_id)
        .where(CampaignReport.campaign_id.in_(campaign_ids))
    ).all()

    latest: dict[str, datetime] = {}
    for campaign_id, published, ingested in rows:
        ts = published or ingested
        if ts is None:
            continue
        ts = _naive(ts)
        if campaign_id not in latest or ts > latest[campaign_id]:
            latest[campaign_id] = ts

    anchor = _naive(now) if now else max(latest.values(), default=None)
    ages: dict[str, float] = {}
    if anchor is not None:
        for campaign_id, ts in latest.items():
            ages[campaign_id] = max((anchor - ts).total_seconds() / 86400.0, 0.0)
    return ages, anchor


def score_campaign_matches(
    session: Session, techniques: set[str], now: datetime | None = None
) -> list[ScoredMatch]:
    """Rank the campaigns whose technique evidence overlaps `techniques`.

    Replaces raw set-overlap: each matched campaign is scored by the rarity of the
    shared techniques, the campaign's freshness, and how firmly it asserts those
    techniques, then sorted by the combined fusion strength (KEV involvement and
    match breadth break ties). Returns an empty list when nothing overlaps.
    """
    if not techniques:
        return []

    edges = session.scalars(
        select(CampaignTechnique).where(CampaignTechnique.technique_id.in_(techniques))
    ).all()
    if not edges:
        return []

    by_campaign: dict[str, list[CampaignTechnique]] = {}
    for edge in edges:
        by_campaign.setdefault(edge.campaign_id, []).append(edge)

    idf = technique_idf(session)
    half_life = get_settings().fusion_recency_half_life_days
    ages, _ = _campaign_ages(session, set(by_campaign), now)

    matches: list[ScoredMatch] = []
    for campaign_id, campaign_edges in by_campaign.items():
        campaign = session.get(Campaign, campaign_id)
        if campaign is None:
            continue
        matched = sorted(e.technique_id for e in campaign_edges)

        # idf defaults to 1.0 for a technique never seen in a report (maximally
        # surprising — it is not part of the feed's common vocabulary).
        specificity = _soft_or([idf.get(t, 1.0) for t in matched])
        corroboration = _soft_or(
            [_corroboration_factor(e.score, e.corroborations) for e in campaign_edges]
        )
        age = ages.get(campaign_id)
        recency = _recency_factor(age, half_life)
        strength = (specificity * recency * corroboration) ** (1.0 / 3.0)

        kev = sorted(
            session.scalars(select(KevEntry.cve_id).where(KevEntry.cve_id.in_(campaign.cve_ids)))
        )
        matches.append(
            ScoredMatch(
                campaign_id=campaign_id,
                cve_ids=campaign.cve_ids,
                kev_cves=kev,
                report_count=campaign.report_count,
                matched_techniques=matched,
                fusion=FusionScore(
                    strength=strength,
                    specificity=specificity,
                    recency=recency,
                    corroboration=corroboration,
                    age_days=age,
                ),
            )
        )

    matches.sort(
        key=lambda m: (m.fusion.strength, len(m.kev_cves), len(m.matched_techniques)),
        reverse=True,
    )
    return matches
