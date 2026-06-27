"""Keyless RSS 2.0 / Atom ingester for threat-intel feeds.

Feeds are untrusted input: parsing goes through defusedxml, which rejects
entity-expansion and external-entity (XXE) constructs outright.
"""

import hashlib
import html
import re
from collections.abc import Mapping
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx
from defusedxml import ElementTree as SafeElementTree
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.config import get_settings
from sentinel.db.models import ThreatReport
from sentinel.logging import get_logger

log = get_logger(__name__)

ATOM_NS = "{http://www.w3.org/2005/Atom}"
_TAG_RE = re.compile(r"<[^>]+>")
# Many publishers (Cloudflare-fronted blogs, CISA) reject the default httpx
# User-Agent and 403 silently; a normal browser UA is required to read them.
_USER_AGENT = "Mozilla/5.0 (compatible; SENTINEL-CTI/0.1; +https://github.com/jaydenOoOo)"


def _clean(text: str | None) -> str | None:
    if not text:
        return None
    return html.unescape(_TAG_RE.sub(" ", text)).strip() or None


def _parse_rfc822(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except ValueError:
        return None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _report_id(link: str | None, fallback: str) -> str:
    digest = hashlib.sha256((link or fallback).encode()).hexdigest()[:24]
    return f"rss:{digest}"


def _parse_rss_items(root: ElementTree.Element, source: str) -> list[ThreatReport]:
    reports = []
    for item in root.findall("./channel/item"):
        title = _clean(item.findtext("title"))
        link = item.findtext("link")
        if not title:
            continue
        reports.append(
            ThreatReport(
                report_id=_report_id(link, item.findtext("guid") or title),
                source=source,
                title=title,
                summary=_clean(item.findtext("description")),
                url=link,
                author=_clean(item.findtext("author")),
                published=_parse_rfc822(item.findtext("pubDate")),
                tags=[c.text for c in item.findall("category") if c.text],
            )
        )
    return reports


def _atom_link(entry: ElementTree.Element) -> str | None:
    """The entry's human-readable article URL.

    An Atom entry can carry several <link>s (rel=alternate/self/edit/replies).
    Blogspot feeds (e.g. Project Zero) list rel="replies" first, so taking the
    first link would store the comments-feed URL instead of the post. Prefer the
    alternate link — or one with no rel, which Atom defines as alternate — and
    fall back to the first link only if there is no better candidate.
    """
    links = entry.findall(f"{ATOM_NS}link")
    for link in links:
        if link.get("rel") in (None, "alternate"):
            return link.get("href")
    return links[0].get("href") if links else None


def _parse_atom_entries(root: ElementTree.Element, source: str) -> list[ThreatReport]:
    reports = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = _clean(entry.findtext(f"{ATOM_NS}title"))
        if not title:
            continue
        link = _atom_link(entry)
        summary = entry.findtext(f"{ATOM_NS}summary") or entry.findtext(f"{ATOM_NS}content")
        published = entry.findtext(f"{ATOM_NS}published") or entry.findtext(f"{ATOM_NS}updated")
        reports.append(
            ThreatReport(
                report_id=_report_id(link, entry.findtext(f"{ATOM_NS}id") or title),
                source=source,
                title=title,
                summary=_clean(summary),
                url=link,
                author=_clean(entry.findtext(f"{ATOM_NS}author/{ATOM_NS}name")),
                published=_parse_iso(published),
                tags=[term for c in entry.findall(f"{ATOM_NS}category") if (term := c.get("term"))],
            )
        )
    return reports


def parse_feed(xml_text: str, source: str = "rss") -> list[ThreatReport]:
    """Parse an RSS 2.0 or Atom feed; `source` is the provenance label per report."""
    root = SafeElementTree.fromstring(xml_text)
    if root.tag == "rss":
        return _parse_rss_items(root, source)
    if root.tag == f"{ATOM_NS}feed":
        return _parse_atom_entries(root, source)
    raise ValueError(f"unsupported feed root element: {root.tag}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=20))
def _fetch_feed(client: httpx.Client, url: str) -> str:
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    return response.text


def fetch_rss_reports(feeds: Mapping[str, str] | None = None) -> list[ThreatReport]:
    """Fetch every configured feed; each report is tagged with its feed's label.

    `feeds` maps provenance label -> URL (defaults to settings.rss_feeds). One
    broken or blocked feed is logged and skipped, never fatal to the run.
    """
    settings = get_settings()
    reports: list[ThreatReport] = []
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/rss+xml, application/atom+xml, */*",
    }
    with httpx.Client(timeout=settings.http_timeout_seconds, headers=headers) as client:
        for source, url in (feeds if feeds is not None else settings.rss_feeds).items():
            try:
                reports.extend(parse_feed(_fetch_feed(client, url), source=source))
            except (httpx.HTTPError, ElementTree.ParseError, ValueError) as exc:
                # One broken feed must not kill the whole ingestion run.
                log.warning("rss_feed_failed", source=source, url=url, error=str(exc))
    return reports
