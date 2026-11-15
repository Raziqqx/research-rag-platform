"""Thin, cached wrappers around sentence-transformers models (CPU)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from rag.config import get_settings


@lru_cache(maxsize=4)
def _bi_encoder(name: str):
    get_settings()  # sets HF_HOME before the first model download
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    get_settings()
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device="cpu")


class Embedder:
    """Callable text -> unit-norm float32 matrix; also usable as chunking.EmbedFn."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _bi_encoder(model_name)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def __call__(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)


def cross_encoder_scores(model_name: str, query: str, passages: list[str]) -> np.ndarray:
    if not passages:
        return np.zeros(0)
    return np.asarray(_cross_encoder(model_name).predict([(query, p) for p in passages]))
