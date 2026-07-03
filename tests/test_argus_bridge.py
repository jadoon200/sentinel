import httpx
import respx

from sentinel.bridge.argus import ArgusBridge, osint_context
from sentinel.config import Settings

# ARGUS's /retrieve returns source-rated OSINT evidence (its EvidenceOut schema).
_OSINT = [
    {
        "doc_id": "reuters.com:1",
        "title": "Edge VPN exploited amid regional tensions",
        "source": "reuters.com",
        "reliability": "B",
        "credibility": 2,
        "rating": "B2",
        "summary": "Reporting links the exploitation wave to a regional dispute.",
        "published": "2026-06-01",
        "url": "https://reuters.com/x",
    }
]


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_osint_context_disabled_without_url() -> None:
    assert osint_context("anything", _settings(argus_api_url="")) == []


@respx.mock
def test_bridge_maps_argus_retrieve_to_rated_osint() -> None:
    respx.get("http://argus.test/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    retrieve = respx.post("http://argus.test/retrieve").mock(
        return_value=httpx.Response(200, json=_OSINT)
    )

    bridge = ArgusBridge("http://argus.test")
    assert bridge.available() is True
    items = bridge.osint_context("edge vpn exploitation", limit=5)

    assert len(items) == 1
    item = items[0]
    assert item.doc_id == "reuters.com:1"
    assert item.reliability == "B" and item.rating == "B2" and item.credibility == 2
    assert "regional" in (item.summary or "")
    # the query (the campaign's subject) and k are passed through to ARGUS
    import json

    body = json.loads(retrieve.calls.last.request.content)
    assert body["query"] == "edge vpn exploitation" and body["k"] == 5


@respx.mock
def test_osint_context_end_to_end() -> None:
    # No /health pre-flight: retrieval is a single round-trip.
    respx.post("http://argus.test/retrieve").mock(return_value=httpx.Response(200, json=_OSINT))
    items = osint_context("q", _settings(argus_api_url="http://argus.test"))
    assert [i.doc_id for i in items] == ["reuters.com:1"]


@respx.mock
def test_osint_context_empty_when_unreachable() -> None:
    respx.post("http://argus.test/retrieve").mock(side_effect=httpx.ConnectError("down"))
    assert osint_context("q", _settings(argus_api_url="http://argus.test")) == []


@respx.mock
def test_osint_context_empty_on_error_response() -> None:
    respx.post("http://argus.test/retrieve").mock(return_value=httpx.Response(503))
    assert osint_context("q", _settings(argus_api_url="http://argus.test")) == []
