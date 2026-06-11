"""Client for the MITRE ATT&CK enterprise STIX 2.1 catalog (free, GitHub-hosted)."""

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.config import get_settings
from sentinel.db.models import AttackTechnique


def _external_reference(obj: dict[str, Any]) -> dict[str, Any] | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return dict(ref)
    return None


def _tactics(obj: dict[str, Any]) -> list[str]:
    return [
        phase["phase_name"]
        for phase in obj.get("kill_chain_phases", [])
        if phase.get("kill_chain_name") == "mitre-attack"
    ]


def parse_techniques(bundle: dict[str, Any]) -> list[AttackTechnique]:
    """Extract active techniques (incl. sub-techniques) from a STIX bundle."""
    techniques = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        ref = _external_reference(obj)
        if ref is None or "external_id" not in ref:
            continue
        techniques.append(
            AttackTechnique(
                technique_id=ref["external_id"],
                name=obj.get("name", ""),
                description=obj.get("description"),
                tactics=_tactics(obj),
                platforms=obj.get("x_mitre_platforms"),
                is_subtechnique=bool(obj.get("x_mitre_is_subtechnique", False)),
                url=ref.get("url"),
                stix_id=obj.get("id"),
            )
        )
    return techniques


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30))
def fetch_attack_techniques() -> list[AttackTechnique]:
    settings = get_settings()
    # The enterprise bundle is ~40 MB; allow a generous read timeout.
    with httpx.Client(timeout=httpx.Timeout(settings.http_timeout_seconds, read=120.0)) as client:
        response = client.get(settings.attack_stix_url)
        response.raise_for_status()
        bundle = response.json()
    return parse_techniques(bundle)
