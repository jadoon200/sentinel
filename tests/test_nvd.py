from typing import Any

import respx
from httpx import Response

from sentinel.config import get_settings
from sentinel.ingest.nvd import fetch_recent_cves


def _nvd_payload(cve_id: str, total: int, per_page: int) -> dict[str, Any]:
    return {
        "resultsPerPage": per_page,
        "startIndex": 0,
        "totalResults": total,
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "published": "2026-06-01T10:00:00.000",
                    "lastModified": "2026-06-08T12:00:00.000",
                    "descriptions": [
                        {"lang": "en", "value": "Remote code execution in ExampleD."},
                        {"lang": "es", "value": "Otra descripción."},
                    ],
                    "metrics": {
                        "cvssMetricV31": [
                            {
                                "cvssData": {"baseScore": 9.8, "baseSeverity": "CRITICAL"},
                            }
                        ]
                    },
                }
            }
        ],
    }


@respx.mock
def test_fetch_recent_cves_parses_fields() -> None:
    url = get_settings().nvd_api_url
    respx.get(url).mock(return_value=Response(200, json=_nvd_payload("CVE-2026-0001", 1, 1)))

    vulns = list(fetch_recent_cves(days=7))

    assert len(vulns) == 1
    vuln = vulns[0]
    assert vuln.cve_id == "CVE-2026-0001"
    assert vuln.cvss_score == 9.8
    assert vuln.cvss_severity == "CRITICAL"
    assert vuln.description is not None and "Remote code execution" in vuln.description
    assert vuln.published is not None and vuln.published.year == 2026


@respx.mock
def test_fetch_recent_cves_pages_through_results() -> None:
    url = get_settings().nvd_api_url
    page1 = _nvd_payload("CVE-2026-0001", total=2, per_page=1)
    page2 = _nvd_payload("CVE-2026-0002", total=2, per_page=1)
    page2["startIndex"] = 1
    route = respx.get(url)
    route.side_effect = [Response(200, json=page1), Response(200, json=page2)]

    vulns = list(fetch_recent_cves(days=7))

    assert [v.cve_id for v in vulns] == ["CVE-2026-0001", "CVE-2026-0002"]
    assert route.call_count == 2
