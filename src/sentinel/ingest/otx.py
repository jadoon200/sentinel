"""Client for AlienVault OTX subscribed pulses (free account API key required)."""

from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.config import get_settings
from sentinel.db.models import ThreatReport


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _attack_ids(pulse: dict[str, Any]) -> list[str]:
    # Older pulses carry plain strings, newer ones {"id": ..., "name": ...}.
    ids = []
    for item in pulse.get("attack_ids", []):
        ids.append(item["id"] if isinstance(item, dict) else str(item))
    return ids


def _parse_pulse(pulse: dict[str, Any]) -> ThreatReport:
    raw = {k: v for k, v in pulse.items() if k != "indicators"}  # indicators are bulky
    return ThreatReport(
        report_id=f"otx:{pulse['id']}",
        source="otx",
        title=pulse.get("name", ""),
        summary=pulse.get("description") or None,
        url=f"https://otx.alienvault.com/pulse/{pulse['id']}",
        author=pulse.get("author_name"),
        published=_parse_datetime(pulse.get("created")),
        tags=pulse.get("tags"),
        attack_ids=_attack_ids(pulse),
        raw=raw,
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30))
def _fetch_page(client: httpx.Client, url: str, params: dict[str, Any] | None) -> dict[str, Any]:
    response = client.get(url, params=params)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload


def fetch_otx_pulses(api_key: str | None = None) -> list[ThreatReport]:
    settings = get_settings()
    key = api_key or settings.otx_api_key
    if not key:
        raise ValueError("OTX API key not configured (SENTINEL_OTX_API_KEY)")

    reports: list[ThreatReport] = []
    url: str | None = settings.otx_api_url
    params: dict[str, Any] | None = {"limit": settings.otx_page_limit}
    with httpx.Client(
        timeout=settings.http_timeout_seconds, headers={"X-OTX-API-KEY": key}
    ) as client:
        for _ in range(settings.otx_max_pages):
            if url is None:
                break
            payload = _fetch_page(client, url, params)
            reports.extend(_parse_pulse(p) for p in payload.get("results", []))
            url = payload.get("next") or None
            params = None  # the `next` URL already carries pagination params
    return reports
