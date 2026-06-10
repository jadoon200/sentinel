from datetime import date

import respx
from httpx import Response

from sentinel.config import get_settings
from sentinel.ingest.kev import fetch_kev_catalog

KEV_PAYLOAD = {
    "title": "CISA Catalog of Known Exploited Vulnerabilities",
    "count": 1,
    "vulnerabilities": [
        {
            "cveID": "CVE-2026-1234",
            "vendorProject": "ExampleCorp",
            "product": "ExampleServer",
            "vulnerabilityName": "ExampleServer Auth Bypass",
            "dateAdded": "2026-06-01",
            "shortDescription": "Authentication bypass allowing remote access.",
            "knownRansomwareCampaignUse": "Known",
            "dueDate": "2026-06-22",
        }
    ],
}


@respx.mock
def test_fetch_kev_catalog_parses_entries() -> None:
    respx.get(get_settings().kev_url).mock(return_value=Response(200, json=KEV_PAYLOAD))

    entries = fetch_kev_catalog()

    assert len(entries) == 1
    entry = entries[0]
    assert entry.cve_id == "CVE-2026-1234"
    assert entry.vendor_project == "ExampleCorp"
    assert entry.known_ransomware_use == "Known"
    assert entry.date_added == date(2026, 6, 1)
    assert entry.due_date == date(2026, 6, 22)
