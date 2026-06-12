"""Tiny BM25 for hybrid lexical+dense retrieval (stdlib + numpy only).

Dense embeddings miss exact identifiers (tool names, registry keys, API
names) that CTI sentences and technique docs share verbatim; BM25 catches
those. Fused with dense ranks via reciprocal-rank fusion in the mapper.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in doc_tokens], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if len(docs) else 1.0
        self.n_docs = len(docs)
        # term -> [(doc index, term frequency)]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, tokens in enumerate(doc_tokens):
            for word, freq in Counter(tokens).items():
                self.postings.setdefault(word, []).append((i, freq))
        self.idf = {
            word: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for word, p in self.postings.items()
        }

    def scores(self, query: str) -> NDArray[np.float64]:
        out = np.zeros(self.n_docs)
        for word in set(tokenize(query)):
            postings = self.postings.get(word)
            if postings is None:
                continue
            idf = self.idf[word]
            for i, freq in postings:
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * freq * (self.k1 + 1) / denom
        return out


def reciprocal_rank_fusion(
    score_lists: Sequence[NDArray[np.float64]], k: int = 60
) -> NDArray[np.float64]:
    """RRF: scale-free fusion of rankings — robust without score calibration."""
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[np.argsort(-scores)] = np.arange(len(scores))
        fused += 1.0 / (k + ranks + 1)
    return fused
