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

from sentinel.db.models import Alert, Campaign, CampaignTechnique, KevEntry


@dataclass
class CampaignLink:
    campaign_id: str
    matched_techniques: list[str]
    report_count: int
    kev_cves: list[str]


@dataclass
class HostThreat:
    host: str
    risk: int
    detectors: list[str]
    techniques: list[str]
    predicted_labels: list[str]
    true_labels: list[str]
    alert_count: int
    fused: list[CampaignLink] = field(default_factory=list)
    simulated: bool = False


# Transparent risk rule. Each input is bounded and named so the score is
# explainable on the dashboard rather than a black box:
#   detector agreement  — how many of the 4 detectors independently flagged the
#                         host (the ensemble's strongest signal)
#   severity            — the host's peak supervised confidence (0..1)
#   intel fusion        — bonus if the host's techniques match a CTI campaign,
#                         larger if that campaign involves KEV (exploited) CVEs
_DETECTOR_WEIGHT = 14  # per distinct detector, capped at 4 -> 56
_SEVERITY_WEIGHT = 24
_FUSION_BONUS = 14
_KEV_BONUS = 6


def _risk(n_detectors: int, severity: float, fused: list[CampaignLink]) -> int:
    score = min(n_detectors, 4) * _DETECTOR_WEIGHT + severity * _SEVERITY_WEIGHT
    if fused:
        score += _FUSION_BONUS
        if any(link.kev_cves for link in fused):
            score += _KEV_BONUS
    return int(min(round(score), 99))


def _campaign_links(session: Session, techniques: set[str]) -> list[CampaignLink]:
    if not techniques:
        return []
    edges = session.scalars(
        select(CampaignTechnique).where(CampaignTechnique.technique_id.in_(techniques))
    ).all()
    by_campaign: dict[str, list[str]] = {}
    for edge in edges:
        by_campaign.setdefault(edge.campaign_id, []).append(edge.technique_id)
    links = []
    for campaign_id, matched in by_campaign.items():
        campaign = session.get(Campaign, campaign_id)
        if campaign is None:
            continue
        kev = sorted(
            session.scalars(select(KevEntry.cve_id).where(KevEntry.cve_id.in_(campaign.cve_ids)))
        )
        links.append(
            CampaignLink(
                campaign_id=campaign_id,
                matched_techniques=sorted(matched),
                report_count=campaign.report_count,
                kev_cves=kev,
            )
        )
    links.sort(key=lambda link: (len(link.kev_cves), len(link.matched_techniques)), reverse=True)
    return links


def host_threats(session: Session, include_simulated: bool = False) -> list[HostThreat]:
    """Group alerts by source host into ranked, intel-fused threats."""
    query = select(Alert).where(Alert.source_host.is_not(None))
    if not include_simulated:
        query = query.where(Alert.simulated.is_(False))

    grouped: dict[str, list[Alert]] = {}
    for alert in session.scalars(query):
        grouped.setdefault(alert.source_host or "", []).append(alert)

    threats = []
    for host, alerts in grouped.items():
        detectors = sorted({a.model for a in alerts})
        techniques = sorted({t for a in alerts for t in (a.techniques or [])})
        severity = max((a.score for a in alerts if a.model == "lightgbm-multiclass"), default=0.0)
        fused = _campaign_links(session, set(techniques))
        threats.append(
            HostThreat(
                host=host,
                risk=_risk(len(detectors), severity, fused),
                detectors=detectors,
                techniques=techniques,
                predicted_labels=sorted({a.predicted_label for a in alerts if a.predicted_label}),
                true_labels=sorted({a.true_label for a in alerts if a.true_label}),
                alert_count=len(alerts),
                fused=fused,
                simulated=any(a.simulated for a in alerts),
            )
        )
    threats.sort(key=lambda t: t.risk, reverse=True)
    return threats
