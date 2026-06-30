"""OSINT-fusion bridge — read-only client for the sibling ARGUS workbench.

The mirror of ARGUS's SENTINEL bridge, completing the loop into a two-way all-source
fusion: when ``SENTINEL_ARGUS_API_URL`` is set, a cyber campaign can be enriched with the
**open-source** picture relevant to it — ARGUS's hybrid OSINT retrieval (``POST /retrieve``),
returning source-rated evidence the analyst can read alongside the cyber graph. SENTINEL only
ever READS from ARGUS; the open-source corpus stays ARGUS's system of record. Disabled
(returns nothing) when the URL is unset or ARGUS is unreachable, so the bridge never breaks a
route.
"""

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from sentinel.config import Settings, get_settings
from sentinel.logging import get_logger

log = get_logger(__name__)


class OsintItem(BaseModel):
    """A rated open-source evidence item from ARGUS (mirrors ARGUS's EvidenceOut)."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    title: str
    source: str
    reliability: str  # NATO Admiralty source reliability A-F
    credibility: int | None = None  # Admiralty information credibility 1-6
    rating: str = ""  # compact Admiralty code, e.g. "B2"
    summary: str | None = None
    published: str | None = None
    url: str | None = None


class ArgusBridge:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._url = base_url.rstrip("/")
        self._timeout = timeout

    def available(self) -> bool:
        try:
            httpx.get(f"{self._url}/health", timeout=self._timeout).raise_for_status()
            return True
        except (httpx.HTTPError, ValueError):
            return False

    def osint_context(self, query: str, limit: int = 5) -> list[OsintItem]:
        """ARGUS's top open-source evidence for `query`, as rated items, or [] on failure."""
        try:
            resp = httpx.post(
                f"{self._url}/retrieve",
                json={"query": query[:4000], "k": limit},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data: Any = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("argus_bridge_failed", error=str(exc))
            return []
        rows = data if isinstance(data, list) else []
        items: list[OsintItem] = []
        for row in rows[:limit]:
            if isinstance(row, dict) and row.get("doc_id"):
                try:
                    items.append(OsintItem.model_validate(row))
                except ValueError:
                    continue
        return items


def osint_context(query: str, settings: Settings | None = None, limit: int = 5) -> list[OsintItem]:
    """ARGUS OSINT relevant to `query` as rated evidence, or [] when the bridge is off."""
    s = settings or get_settings()
    if not s.argus_api_url:
        return []
    bridge = ArgusBridge(s.argus_api_url, s.http_timeout_seconds)
    if not bridge.available():
        log.warning("argus_bridge_unreachable", url=s.argus_api_url)
        return []
    return bridge.osint_context(query, limit)
