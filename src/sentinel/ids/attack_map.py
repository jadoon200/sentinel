"""Map CIC-IDS2017 attack labels to ATT&CK techniques.

This is what lets IDS alerts join the same knowledge graph as CTI reports:
an alert for a flow classified as e.g. PortScan carries T1046, so the fusion
layer can correlate it with campaigns whose reports mention the technique.

Mappings are curated, not learned — each one is defensible from how the
attack was actually executed in the CIC-IDS2017 testbed.
"""

# Base label (without the corrected dataset's " - Attempted" suffix) → techniques.
_LABEL_TECHNIQUES: dict[str, list[str]] = {
    "BENIGN": [],
    # Patator brute-forcing FTP/SSH credentials
    "FTP-PATATOR": ["T1110"],
    "SSH-PATATOR": ["T1110"],
    # Application-layer DoS tools against the web server
    "DOS HULK": ["T1499"],
    "DOS GOLDENEYE": ["T1499"],
    "DOS SLOWLORIS": ["T1499"],
    "DOS SLOWHTTPTEST": ["T1499"],
    "DDOS": ["T1498"],
    # OpenSSL Heartbleed exploitation of the public-facing service
    "HEARTBLEED": ["T1190"],
    # DVWA web attacks (server-side exploitation of a public-facing app)
    "WEB ATTACK - BRUTE FORCE": ["T1110"],
    "WEB ATTACK - XSS": ["T1190"],
    "WEB ATTACK - SQL INJECTION": ["T1190"],
    # Meterpreter dropped via infected file download, then scans inside
    "INFILTRATION": ["T1203", "T1105"],
    "INFILTRATION - PORTSCAN": ["T1046"],
    # ARES botnet HTTP command-and-control
    "BOTNET": ["T1071.001"],
    "BOT": ["T1071.001"],
    # Nmap sweeps from the firewall host
    "PORTSCAN": ["T1046"],
}


def _normalize(label: str) -> str:
    # Original CSVs use \x96 / en dash in web attack labels; corrected ones use "-".
    cleaned = label.replace("\x96", "-").replace("\u2013", "-").strip().upper()
    cleaned = " ".join(cleaned.split())
    return cleaned.removesuffix(" - ATTEMPTED")


def techniques_for_label(label: str) -> list[str]:
    """ATT&CK technique IDs for a CIC-IDS2017 flow label (empty for benign/unknown)."""
    return _LABEL_TECHNIQUES.get(_normalize(label), [])
