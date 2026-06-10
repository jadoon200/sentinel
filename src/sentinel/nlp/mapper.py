"""Map CTI text to ATT&CK techniques via embedding retrieval + optional reranking.

Instead of a multi-label classifier capped at the most common techniques, the
mapper embeds the full technique catalog and retrieves nearest techniques for a
piece of text (bi-encoder), optionally reranked by a cross-encoder. Evidence
from multiple texts (sentences, reports of one campaign) is corroborated with
`aggregate_matches`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinel.db.models import AttackTechnique


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]: ...


class PairScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(frozen=True)
class TechniqueDoc:
    technique_id: str
    name: str
    text: str


@dataclass(frozen=True)
class TechniqueMatch:
    technique_id: str
    name: str
    score: float


@dataclass(frozen=True)
class CorroboratedMatch:
    technique_id: str
    name: str
    corroborations: int
    score: float


def technique_doc(technique: AttackTechnique, max_chars: int = 2000) -> TechniqueDoc:
    description = (technique.description or "")[:max_chars]
    return TechniqueDoc(
        technique_id=technique.technique_id,
        name=technique.name,
        text=f"{technique.name}. {description}",
    )


def load_technique_docs(session: Session) -> list[TechniqueDoc]:
    techniques = session.scalars(select(AttackTechnique)).all()
    return [technique_doc(t) for t in techniques]


def _normalize(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / np.maximum(norms, 1e-12))


class TechniqueMapper:
    """Retrieve (and optionally rerank) ATT&CK techniques for free text."""

    def __init__(
        self,
        docs: Sequence[TechniqueDoc],
        encoder: TextEncoder,
        reranker: PairScorer | None = None,
    ) -> None:
        if not docs:
            raise ValueError("technique catalog is empty — run the ATT&CK ingester first")
        self._docs = list(docs)
        self._encoder = encoder
        self._reranker = reranker
        self._index: NDArray[np.floating] | None = None

    def _ensure_index(self) -> NDArray[np.floating]:
        if self._index is None:
            embeddings = self._encoder.encode([doc.text for doc in self._docs])
            self._index = _normalize(np.asarray(embeddings))
        return self._index

    def map_text(self, text: str, top_k: int = 5, candidates: int = 20) -> list[TechniqueMatch]:
        index = self._ensure_index()
        query = _normalize(np.asarray(self._encoder.encode([text])))[0]
        similarities = index @ query

        candidate_count = max(top_k, candidates) if self._reranker else top_k
        order = np.argsort(similarities)[::-1][:candidate_count]
        matches = [
            TechniqueMatch(
                technique_id=self._docs[i].technique_id,
                name=self._docs[i].name,
                score=float(similarities[i]),
            )
            for i in order
        ]

        if self._reranker is not None:
            pairs = [(text, self._docs[i].text) for i in order]
            scores = self._reranker.score(pairs)
            matches = [
                TechniqueMatch(m.technique_id, m.name, float(s))
                for m, s in zip(matches, scores, strict=True)
            ]
            matches.sort(key=lambda m: m.score, reverse=True)

        return matches[:top_k]


def aggregate_matches(
    per_text_matches: Iterable[Sequence[TechniqueMatch]],
) -> list[CorroboratedMatch]:
    """Corroborate technique evidence across texts (sentences or campaign reports).

    Techniques seen in more texts rank higher; mean score breaks ties. Multi-report
    aggregation is the cheapest known accuracy win for technique extraction
    (~+26% F1, arXiv:2604.07470).
    """
    counts: dict[str, int] = {}
    scores: dict[str, list[float]] = {}
    names: dict[str, str] = {}
    for matches in per_text_matches:
        for match in matches:
            counts[match.technique_id] = counts.get(match.technique_id, 0) + 1
            scores.setdefault(match.technique_id, []).append(match.score)
            names[match.technique_id] = match.name

    aggregated = [
        CorroboratedMatch(
            technique_id=technique_id,
            name=names[technique_id],
            corroborations=count,
            score=sum(scores[technique_id]) / count,
        )
        for technique_id, count in counts.items()
    ]
    aggregated.sort(key=lambda m: (m.corroborations, m.score), reverse=True)
    return aggregated
