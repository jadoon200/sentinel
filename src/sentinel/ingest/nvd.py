"""Client for the NVD CVE 2.0 API.

Free to use; an (also free) API key raises the rate limit from 5 to 50
requests per 30 seconds. Docs: https://nvd.nist.gov/developers/vulnerabilities
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.config import get_settings
from sentinel.db.models import Vulnerability
from sentinel.logging import get_logger

log = get_logger(__name__)


def _parse_cve(item: dict[str, Any]) -> Vulnerability:
    cve = item["cve"]
    description = next(
        (d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), None
    )

    score: float | None = None
    severity: str | None = None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if metrics.get(key):
            data = metrics[key][0]["cvssData"]
            score = data.get("baseScore")
            severity = data.get("baseSeverity") or metrics[key][0].get("baseSeverity")
            break

    def _ts(field: str) -> datetime | None:
        value = cve.get(field)
        return datetime.fromisoformat(value).replace(tzinfo=UTC) if value else None

    return Vulnerability(
        cve_id=cve["id"],
        description=description,
        cvss_score=score,
        cvss_severity=severity,
        published=_ts("published"),
        last_modified=_ts("lastModified"),
        raw=cve,
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30))
def _fetch_page(client: httpx.Client, params: dict[str, Any]) -> dict[str, Any]:
    response = client.get(get_settings().nvd_api_url, params=params)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def fetch_recent_cves(days: int = 7) -> Iterator[Vulnerability]:
    """Yield CVEs modified in the last `days` days, paging through the API."""
    settings = get_settings()
    end = datetime.now(UTC)
    start = end - timedelta(days=days)

    headers = {"apiKey": settings.nvd_api_key} if settings.nvd_api_key else {}
    params: dict[str, Any] = {
        "lastModStartDate": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "lastModEndDate": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "resultsPerPage": settings.nvd_page_size,
        "startIndex": 0,
    }

    with httpx.Client(timeout=settings.http_timeout_seconds, headers=headers) as client:
        while True:
            payload = _fetch_page(client, params)
            for item in payload.get("vulnerabilities", []):
                yield _parse_cve(item)

            params["startIndex"] += payload.get("resultsPerPage", 0)
            if params["startIndex"] >= payload.get("totalResults", 0) or not payload.get(
                "vulnerabilities"
            ):
                break
