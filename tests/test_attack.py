import respx
from httpx import Response

from sentinel.config import get_settings
from sentinel.ingest.attack import fetch_attack_techniques, parse_techniques

STIX_BUNDLE = {
    "type": "bundle",
    "objects": [
        {
            "type": "attack-pattern",
            "id": "attack-pattern--0001",
            "name": "PowerShell",
            "description": "Adversaries may abuse PowerShell commands and scripts.",
            "x_mitre_is_subtechnique": True,
            "x_mitre_platforms": ["Windows"],
            "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "execution"}],
            "external_references": [
                {
                    "source_name": "mitre-attack",
                    "external_id": "T1059.001",
                    "url": "https://attack.mitre.org/techniques/T1059/001",
                }
            ],
        },
        {
            "type": "attack-pattern",
            "id": "attack-pattern--0002",
            "name": "Old Technique",
            "x_mitre_deprecated": True,
            "external_references": [{"source_name": "mitre-attack", "external_id": "T9999"}],
        },
        {"type": "intrusion-set", "id": "intrusion-set--0003", "name": "APT-Example"},
    ],
}


def test_parse_techniques_extracts_active_attack_patterns() -> None:
    techniques = parse_techniques(STIX_BUNDLE)

    assert len(techniques) == 1
    technique = techniques[0]
    assert technique.technique_id == "T1059.001"
    assert technique.name == "PowerShell"
    assert technique.is_subtechnique is True
    assert technique.tactics == ["execution"]
    assert technique.platforms == ["Windows"]
    assert technique.url == "https://attack.mitre.org/techniques/T1059/001"


@respx.mock
def test_fetch_attack_techniques_hits_stix_url() -> None:
    respx.get(get_settings().attack_stix_url).mock(return_value=Response(200, json=STIX_BUNDLE))

    techniques = fetch_attack_techniques()

    assert [t.technique_id for t in techniques] == ["T1059.001"]
