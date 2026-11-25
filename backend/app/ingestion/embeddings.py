"""Sentence embeddings with FastEmbed (ONNX Runtime) — no PyTorch.

Same BAAI/bge-small-en-v1.5 model as before, served as an optimised ONNX file
(~67 MB). Total API memory drops from ~1 GB to a few hundred MB, which is what lets
the API run on 512 MB free hosting tiers.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class FastEmbedEmbeddings(Embeddings):
    """LangChain wrapper around fastembed.TextEmbedding.

    Vectors are L2-normalised, so cosine similarity in Chroma lands in a predictable
    0..1 range (the confidence thresholds depend on that).
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        batch_size: int = 32,
        threads: int | None = None,
    ):
        from fastembed import TextEmbedding  # lazy: tests use fake embeddings

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    @staticmethod
    def _normalise(vectors) -> list[list[float]]:
        arr = np.asarray(list(vectors), dtype=np.float32)
        if arr.size == 0:
            return []
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._normalise(self._model.embed(texts, batch_size=self.batch_size))

    def embed_query(self, text: str) -> list[float]:
        return self._normalise(self._model.embed([text], batch_size=1))[0]


@lru_cache(maxsize=2)
def get_embeddings(
    model_name: str,
    cache_dir: str | None = None,
    batch_size: int = 32,
    threads: int | None = None,
) -> Embeddings:
    """Load the embedding model once per process (first call downloads it if not cached)."""
    logger.info("Loading embedding model %s (FastEmbed/ONNX)", model_name)
    return FastEmbedEmbeddings(
        model_name, cache_dir=cache_dir, batch_size=batch_size, threads=threads
    )
