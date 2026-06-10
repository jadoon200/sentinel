from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sentinel.nlp.mapper import (
    TechniqueDoc,
    TechniqueMapper,
    TechniqueMatch,
    aggregate_matches,
)

VOCAB = ["powershell", "phishing", "registry", "scheduled", "exfiltration"]

DOCS = [
    TechniqueDoc("T1059.001", "PowerShell", "PowerShell. Abuse of powershell scripts."),
    TechniqueDoc("T1566", "Phishing", "Phishing. Emails with malicious phishing links."),
    TechniqueDoc("T1053", "Scheduled Task", "Scheduled Task. Persistence via scheduled jobs."),
]


class KeywordEncoder:
    """Deterministic bag-of-words embedding over a tiny vocabulary."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([float(lowered.count(word)) for word in VOCAB])
        return np.asarray(rows)


class KeywordOverlapScorer:
    """Scores a pair by shared vocabulary words."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [float(sum(1 for w in VOCAB if w in a.lower() and w in b.lower())) for a, b in pairs]


def test_map_text_retrieves_matching_technique() -> None:
    mapper = TechniqueMapper(DOCS, encoder=KeywordEncoder())

    matches = mapper.map_text("the actor executed obfuscated powershell payloads", top_k=2)

    assert matches[0].technique_id == "T1059.001"
    assert matches[0].score > matches[1].score


def test_reranker_overrides_retrieval_order() -> None:
    class FlippedScorer:
        def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [1.0 if "phishing" in doc.lower() else 0.0 for _, doc in pairs]

    mapper = TechniqueMapper(DOCS, encoder=KeywordEncoder(), reranker=FlippedScorer())

    matches = mapper.map_text("powershell and a phishing email", top_k=1, candidates=3)

    assert matches[0].technique_id == "T1566"


def test_aggregate_matches_prefers_corroborated_techniques() -> None:
    report_a = [TechniqueMatch("T1059.001", "PowerShell", 0.6)]
    report_b = [
        TechniqueMatch("T1059.001", "PowerShell", 0.5),
        TechniqueMatch("T1566", "Phishing", 0.99),
    ]

    aggregated = aggregate_matches([report_a, report_b])

    assert aggregated[0].technique_id == "T1059.001"
    assert aggregated[0].corroborations == 2
    assert aggregated[1].technique_id == "T1566"
