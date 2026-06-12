import pytest
import respx
from httpx import Response

from sentinel.ingest.rss import fetch_rss_reports, parse_feed

RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example Threat Blog</title>
    <item>
      <title>New ransomware exploits &lt;b&gt;VPN&lt;/b&gt; flaw</title>
      <link>https://example.com/post/1</link>
      <description>&lt;p&gt;Actors exploit CVE-2026-0001 for initial access.&lt;/p&gt;</description>
      <pubDate>Tue, 09 Jun 2026 10:00:00 GMT</pubDate>
      <category>ransomware</category>
    </item>
  </channel>
</rss>
"""

ATOM_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Advisories</title>
  <entry>
    <id>urn:adv:2026-42</id>
    <title>Advisory 2026-42</title>
    <link href="https://example.org/adv/42"/>
    <summary>Patch now: actively exploited deserialization bug.</summary>
    <published>2026-06-08T12:00:00Z</published>
    <category term="advisory"/>
    <author><name>CERT Example</name></author>
  </entry>
</feed>
"""


def test_parse_rss_feed() -> None:
    reports = parse_feed(RSS_FEED)

    assert len(reports) == 1
    report = reports[0]
    assert report.report_id.startswith("rss:")
    assert report.title == "New ransomware exploits  VPN  flaw"
    assert report.summary == "Actors exploit CVE-2026-0001 for initial access."
    assert report.url == "https://example.com/post/1"
    assert report.tags == ["ransomware"]
    assert report.published is not None and report.published.year == 2026


def test_parse_atom_feed() -> None:
    reports = parse_feed(ATOM_FEED)

    assert len(reports) == 1
    report = reports[0]
    assert report.title == "Advisory 2026-42"
    assert report.url == "https://example.org/adv/42"
    assert report.author == "CERT Example"
    assert report.tags == ["advisory"]


@respx.mock
def test_fetch_rss_reports_survives_broken_feed() -> None:
    respx.get("https://ok.example/feed").mock(return_value=Response(200, text=RSS_FEED))
    respx.get("https://broken.example/feed").mock(
        return_value=Response(200, text="this is not xml")
    )

    reports = fetch_rss_reports(feeds=["https://broken.example/feed", "https://ok.example/feed"])

    assert [r.title for r in reports] == ["New ransomware exploits  VPN  flaw"]


def test_xxe_entity_payload_is_rejected() -> None:
    evil = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE rss [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        '<rss version="2.0"><channel><item>'
        "<title>&x;</title><link>http://evil.example</link>"
        "</item></channel></rss>"
    )

    # defusedxml exceptions subclass ValueError, so the per-feed error
    # handling in fetch_rss_reports also survives a hostile feed.
    with pytest.raises(ValueError):
        parse_feed(evil)
