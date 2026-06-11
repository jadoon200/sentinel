"""Extract CVE identifiers from free text."""

import re

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)


def extract_cve_ids(text: str) -> list[str]:
    """Return unique CVE IDs (uppercased) in order of first appearance."""
    seen: dict[str, None] = {}
    for match in _CVE_RE.findall(text):
        seen.setdefault(match.upper())
    return list(seen)
