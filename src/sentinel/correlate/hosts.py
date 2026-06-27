"""Roll IDS alerts up into per-host threats — the fusion unit of the dashboard.

A host is not "four alerts from four models"; it is one entity with a rap
sheet. This groups a host's alerts across the detector ensemble, unions their
ATT&CK techniques, and scores the host with a transparent, explainable rule
(no opaque meta-model) so the dashboard can rank threats and join each one
against the CTI campaigns via the technique overlap already used by
`/alerts/{id}/context`.
"""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.correlate.fusion import (
    FusionContext,
    FusionScore,
    build_fusion_context,
    score_with_context,
)
from sentinel.db.models import Alert


@dataclass
class CampaignLink:
    campaign_id: str
    matched_techniques: list[str]
    report_count: int
    kev_cves: list[str]
    fusion: FusionScore


@dataclass
class AlertRef:
    """The strongest detection per detector — the entry points for drilling into
    a single alert's campaign context (`/alerts/{id}/context`)."""

    alert_id: int
    model: str
    score: float
    predicted_label: str | None
    techniques: list[str]


@dataclass
class HostThreat:
    host: str
    risk: int
    detectors: list[str]
    techniques: list[str]
    predicted_labels: list[str]
    true_labels: list[str]
    alert_count: int
    alerts: list[AlertRef] = field(default_factory=list)
    fused: list[CampaignLink] = field(default_factory=list)
    simulated: bool = False


# Transparent risk rule. Each input is bounded and named so the score is
# explainable on the dashboard rather than a black box:
#   detector agreement  — how many of the 5 *flow* detectors independently
#                         flagged the host (the ensemble's strongest signal)
#   severity            — the host's peak calibrated attack probability (0..1)
#   intel fusion        — bonus scaled by the *strength* of the best campaign
#                         correlation (specificity x recency x corroboration),
#                         so a rare, recent, well-evidenced match lifts risk far
#                         more than a coincidental shared tag; a further bump if
#                         that campaign involves KEV (exploited) CVEs
# The five flow-ensemble detectors whose agreement the bonus counts. SQLi is a
# separate application-layer (WAF) modality — surfaced on its own in the UI — so
# it must not substitute for a flow detector in the consensus count.
_FLOW_DETECTORS = frozenset({"lightgbm-multiclass", "autoencoder", "sequence", "profile", "beacon"})
# Detectors whose score is a calibrated [0, 1] attack probability, so it can
# drive the severity term; the anomaly detectors emit unbounded recon/z-scores.
_PROBABILITY_MODELS = frozenset({"lightgbm-multiclass", "sqli"})
_N_DETECTORS = 5  # supervised, autoencoder, sequence, profile, beacon
_DETECTOR_WEIGHT = 11  # per distinct flow detector, capped at 5 -> 55
_SEVERITY_WEIGHT = 24
_FUSION_BONUS = 14
_KEV_BONUS = 6


def _risk(n_detectors: int, severity: float, fused: list[CampaignLink]) -> int:
    score = min(n_detectors, _N_DETECTORS) * _DETECTOR_WEIGHT + severity * _SEVERITY_WEIGHT
    if fused:
        best_strength = max(link.fusion.strength for link in fused)
        score += _FUSION_BONUS * best_strength
        if any(link.kev_cves for link in fused):
            score += _KEV_BONUS
    return int(min(round(score), 99))


def _campaign_links(techniques: set[str], ctx: FusionContext) -> list[CampaignLink]:
    return [
        CampaignLink(
            campaign_id=match.campaign_id,
            matched_techniques=match.matched_techniques,
            report_count=match.report_count,
            kev_cves=match.kev_cves,
            fusion=match.fusion,
        )
        for match in score_with_context(techniques, ctx)
    ]


def _alert_refs(alerts: list[Alert]) -> list[AlertRef]:
    """One representative detection per detector — the host's strongest alert from
    each model — so the UI can drill into each on its own campaign context."""
    best: dict[str, Alert] = {}
    for alert in alerts:
        current = best.get(alert.model)
        if current is None or alert.score > current.score:
            best[alert.model] = alert
    return [
        AlertRef(
            alert_id=a.alert_id,
            model=a.model,
            score=a.score,
            predicted_label=a.predicted_label,
            techniques=a.techniques or [],
        )
        for a in sorted(best.values(), key=lambda a: a.model)
    ]


def host_threats(session: Session, include_simulated: bool = False) -> list[HostThreat]:
    """Group alerts by source host into ranked, intel-fused threats."""
    query = select(Alert).where(Alert.source_host.is_not(None))
    if not include_simulated:
        query = query.where(Alert.simulated.is_(False))

    grouped: dict[str, list[Alert]] = {}
    for alert in session.scalars(query):
        grouped.setdefault(alert.source_host or "", []).append(alert)

    # Build the corpus-wide fusion state once, then score every host against it —
    # one set of queries instead of per-host, and a single shared recency anchor.
    ctx = build_fusion_context(session)

    threats = []
    for host, alerts in grouped.items():
        detectors = sorted({a.model for a in alerts})
        techniques = sorted({t for a in alerts for t in (a.techniques or [])})
        n_flow_detectors = len({a.model for a in alerts if a.model in _FLOW_DETECTORS})
        severity = max((a.score for a in alerts if a.model in _PROBABILITY_MODELS), default=0.0)
        fused = _campaign_links(set(techniques), ctx)
        threats.append(
            HostThreat(
                host=host,
                risk=_risk(n_flow_detectors, severity, fused),
                detectors=detectors,
                techniques=techniques,
                predicted_labels=sorted({a.predicted_label for a in alerts if a.predicted_label}),
                true_labels=sorted({a.true_label for a in alerts if a.true_label}),
                alert_count=len(alerts),
                alerts=_alert_refs(alerts),
                fused=fused,
                simulated=any(a.simulated for a in alerts),
            )
        )
    threats.sort(key=lambda t: t.risk, reverse=True)
    return threats
