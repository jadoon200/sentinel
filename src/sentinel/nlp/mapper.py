"""Map CTI text to ATT&CK techniques via embedding retrieval + optional reranking.

Instead of a multi-label classifier capped at the most common techniques, the
mapper embeds the full technique catalog and retrieves nearest techniques for a
piece of text (bi-encoder), optionally reranked by a cross-encoder. Evidence
from multiple texts (sentences, reports of one campaign) is corroborated with
`aggregate_matches`.
"""

import hashlib
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
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


def technique_doc(
    technique: AttackTechnique,
    max_chars: int = 2000,
    include_procedures: bool = False,
    max_procedures: int = 2,
) -> TechniqueDoc:
    """Build the retrieval document; optionally append real procedure examples.

    Procedure enrichment is benchmark-gated (see docs/EVAL.md) — pass
    include_procedures=True only where the TRAM harness showed a win.
    """
    description = (technique.description or "")[:max_chars]
    text = f"{technique.name}. {description}"
    if include_procedures and technique.procedure_examples:
        examples = " ".join(e[:300] for e in technique.procedure_examples[:max_procedures])
        text = f"{text} Procedures: {examples}"
    return TechniqueDoc(
        technique_id=technique.technique_id,
        name=technique.name,
        text=text,
    )


def load_technique_docs(session: Session, include_procedures: bool = True) -> list[TechniqueDoc]:
    """Procedure enrichment defaults on: +10pp hit@5 with hybrid retrieval (EVAL.md)."""
    techniques = session.scalars(select(AttackTechnique)).all()
    return [technique_doc(t, include_procedures=include_procedures) for t in techniques]


def _normalize(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / np.maximum(norms, 1e-12))


def _embedding_cache_path(cache_dir: Path, model_name: str, docs: Sequence[TechniqueDoc]) -> Path:
    digest = hashlib.sha256(model_name.encode())
    for doc in docs:
        digest.update(b"\x00")
        digest.update(doc.text.encode())
    return cache_dir / f"technique_embeddings-{digest.hexdigest()[:16]}.npz"


class TechniqueMapper:
    """Retrieve (and optionally rerank) ATT&CK techniques for free text."""

    def __init__(
        self,
        docs: Sequence[TechniqueDoc],
        encoder: TextEncoder,
        reranker: PairScorer | None = None,
        cache_dir: Path | None = None,
        model_name: str | None = None,
        lexical: bool = False,
    ) -> None:
        if not docs:
            raise ValueError("technique catalog is empty — run the ATT&CK ingester first")
        if (cache_dir is None) != (model_name is None):
            raise ValueError("cache_dir and model_name must be provided together")
        self._docs = list(docs)
        self._encoder = encoder
        self._reranker = reranker
        self._index: NDArray[np.floating] | None = None
        self._bm25 = None
        if lexical:
            from sentinel.nlp.lexical import BM25

            self._bm25 = BM25([doc.text for doc in self._docs])
        self._cache_path = (
            _embedding_cache_path(cache_dir, model_name, self._docs)
            if cache_dir is not None and model_name is not None
            else None
        )

    def _ensure_index(self) -> NDArray[np.floating]:
        if self._index is None:
            embeddings = self._load_cached_embeddings()
            if embeddings is None:
                embeddings = np.asarray(self._encoder.encode([doc.text for doc in self._docs]))
                self._save_embeddings(embeddings)
            self._index = _normalize(embeddings)
        return self._index

    def _load_cached_embeddings(self) -> NDArray[np.floating] | None:
        if self._cache_path is None or not self._cache_path.exists():
            return None
        try:
            with np.load(self._cache_path) as archive:
                embeddings = np.asarray(archive["embeddings"])
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self._docs):
            return None
        return embeddings

    def _save_embeddings(self, embeddings: NDArray[np.floating]) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._cache_path, embeddings=embeddings)

    def map_text(self, text: str, top_k: int = 5, candidates: int = 20) -> list[TechniqueMatch]:
        index = self._ensure_index()
        query = _normalize(np.asarray(self._encoder.encode([text])))[0]
        cosine = np.asarray(index @ query, dtype=np.float64)
        ranking = cosine
        if self._bm25 is not None:
            from sentinel.nlp.lexical import reciprocal_rank_fusion

            # Rank by fusion, but report the dense cosine: RRF scores are
            # rank-based and carry no absolute confidence, while downstream
            # thresholds (report tagging) are calibrated on the cosine scale.
            ranking = reciprocal_rank_fusion([cosine, self._bm25.scores(text)])

        candidate_count = max(top_k, candidates) if self._reranker else top_k
        order = np.argsort(ranking)[::-1][:candidate_count]
        matches = [
            TechniqueMatch(
                technique_id=self._docs[i].technique_id,
                name=self._docs[i].name,
                score=float(cosine[i]),
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
