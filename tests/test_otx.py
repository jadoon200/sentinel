from datetime import UTC, datetime

import pytest
import respx
from httpx import Response

from sentinel.config import get_settings
from sentinel.ingest.otx import fetch_otx_pulses

PAGE_ONE = {
    "results": [
        {
            "id": "6841a1b2c3d4e5f601234567",
            "name": "APT-Example targets regional telcos",
            "description": "Spearphishing campaign delivering a PowerShell loader.",
            "author_name": "alienvault",
            "created": "2026-06-01T08:30:00.123456",
            "tags": ["apt", "telco"],
            "attack_ids": [{"id": "T1566.001", "name": "Spearphishing Attachment"}],
            "indicators": [{"indicator": "203.0.113.7", "type": "IPv4"}],
        }
    ],
    "next": None,
}


@respx.mock
def test_fetch_otx_pulses_parses_reports() -> None:
    respx.get(get_settings().otx_api_url).mock(return_value=Response(200, json=PAGE_ONE))

    reports = fetch_otx_pulses(api_key="test-key")

    assert len(reports) == 1
    report = reports[0]
    assert report.report_id == "otx:6841a1b2c3d4e5f601234567"
    assert report.source == "otx"
    assert report.title == "APT-Example targets regional telcos"
    assert report.attack_ids == ["T1566.001"]
    assert report.published == datetime(2026, 6, 1, 8, 30, 0, 123456, tzinfo=UTC)
    assert report.raw is not None and "indicators" not in report.raw


@respx.mock
def test_fetch_otx_pulses_follows_pagination() -> None:
    settings = get_settings()
    second_url = f"{settings.otx_api_url}?limit=50&page=2"
    page_one = {
        "results": [{"id": "a" * 24, "name": "first", "attack_ids": ["T1059"]}],
        "next": second_url,
    }
    page_two = {"results": [{"id": "b" * 24, "name": "second"}], "next": None}
    route = respx.get(settings.otx_api_url)
    route.side_effect = [Response(200, json=page_one), Response(200, json=page_two)]

    reports = fetch_otx_pulses(api_key="test-key")

    assert [r.title for r in reports] == ["first", "second"]
    assert reports[0].attack_ids == ["T1059"]


def test_fetch_otx_pulses_requires_key() -> None:
    with pytest.raises(ValueError, match="OTX API key"):
        fetch_otx_pulses()
