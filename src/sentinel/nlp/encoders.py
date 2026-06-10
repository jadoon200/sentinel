"""SecureBERT 2.0 adapters (Apache-2.0, run locally via sentence-transformers).

Heavy ML imports are deferred to construction so that importing sentinel.nlp
stays cheap and unit tests can use fake encoders.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sentinel.config import get_settings


class BiEncoder:
    """Sentence-embedding encoder for retrieval (cisco-ai/SecureBERT2.0-biencoder)."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name or get_settings().nlp_bi_encoder_model)

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        return np.asarray(self._model.encode(list(texts), convert_to_numpy=True))


class CrossEncoderScorer:
    """Pairwise relevance scorer for reranking (cisco-ai/SecureBERT2.0-cross_encoder)."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name or get_settings().nlp_cross_encoder_model)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self._model.predict(list(pairs))]
