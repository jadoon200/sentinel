"""Keyless RSS 2.0 / Atom ingester for threat-intel feeds (stdlib XML, no extra deps)."""

import hashlib
import html
import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.config import get_settings
from sentinel.db.models import ThreatReport
from sentinel.logging import get_logger

log = get_logger(__name__)

ATOM_NS = "{http://www.w3.org/2005/Atom}"
_TAG_RE = re.compile(r"<[^>]+>")


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


def _parse_rss_items(root: ElementTree.Element) -> list[ThreatReport]:
    reports = []
    for item in root.findall("./channel/item"):
        title = _clean(item.findtext("title"))
        link = item.findtext("link")
        if not title:
            continue
        reports.append(
            ThreatReport(
                report_id=_report_id(link, item.findtext("guid") or title),
                source="rss",
                title=title,
                summary=_clean(item.findtext("description")),
                url=link,
                author=_clean(item.findtext("author")),
                published=_parse_rfc822(item.findtext("pubDate")),
                tags=[c.text for c in item.findall("category") if c.text],
            )
        )
    return reports


def _parse_atom_entries(root: ElementTree.Element) -> list[ThreatReport]:
    reports = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = _clean(entry.findtext(f"{ATOM_NS}title"))
        if not title:
            continue
        link_el = entry.find(f"{ATOM_NS}link")
        link = link_el.get("href") if link_el is not None else None
        summary = entry.findtext(f"{ATOM_NS}summary") or entry.findtext(f"{ATOM_NS}content")
        published = entry.findtext(f"{ATOM_NS}published") or entry.findtext(f"{ATOM_NS}updated")
        reports.append(
            ThreatReport(
                report_id=_report_id(link, entry.findtext(f"{ATOM_NS}id") or title),
                source="rss",
                title=title,
                summary=_clean(summary),
                url=link,
                author=_clean(entry.findtext(f"{ATOM_NS}author/{ATOM_NS}name")),
                published=_parse_iso(published),
                tags=[term for c in entry.findall(f"{ATOM_NS}category") if (term := c.get("term"))],
            )
        )
    return reports


def parse_feed(xml_text: str) -> list[ThreatReport]:
    root = ElementTree.fromstring(xml_text)
    if root.tag == "rss":
        return _parse_rss_items(root)
    if root.tag == f"{ATOM_NS}feed":
        return _parse_atom_entries(root)
    raise ValueError(f"unsupported feed root element: {root.tag}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=20))
def _fetch_feed(client: httpx.Client, url: str) -> str:
    response = client.get(url, follow_redirects=True)
    response.raise_for_status()
    return response.text


def fetch_rss_reports(feeds: list[str] | None = None) -> list[ThreatReport]:
    settings = get_settings()
    reports: list[ThreatReport] = []
    with httpx.Client(timeout=settings.http_timeout_seconds) as client:
        for url in feeds if feeds is not None else settings.rss_feeds:
            try:
                reports.extend(parse_feed(_fetch_feed(client, url)))
            except (httpx.HTTPError, ElementTree.ParseError, ValueError) as exc:
                # One broken feed must not kill the whole ingestion run.
                log.warning("rss_feed_failed", url=url, error=str(exc))
    return reports
