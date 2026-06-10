"""Client for the CISA Known Exploited Vulnerabilities catalog (free public JSON)."""

from datetime import date
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.config import get_settings
from sentinel.db.models import KevEntry


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _parse_entry(item: dict[str, Any]) -> KevEntry:
    return KevEntry(
        cve_id=item["cveID"],
        vendor_project=item.get("vendorProject"),
        product=item.get("product"),
        vulnerability_name=item.get("vulnerabilityName"),
        short_description=item.get("shortDescription"),
        known_ransomware_use=item.get("knownRansomwareCampaignUse"),
        date_added=_parse_date(item.get("dateAdded")),
        due_date=_parse_date(item.get("dueDate")),
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30))
def fetch_kev_catalog() -> list[KevEntry]:
    settings = get_settings()
    with httpx.Client(timeout=settings.http_timeout_seconds) as client:
        response = client.get(settings.kev_url)
        response.raise_for_status()
        payload = response.json()
    return [_parse_entry(item) for item in payload.get("vulnerabilities", [])]
