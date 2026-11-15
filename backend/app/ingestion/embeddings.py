"""HuggingFace sentence-embedding model, loaded once per process."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_embeddings(model_name: str, device: str = "cpu", batch_size: int = 32) -> Embeddings:
    """Return a cached embedding model.

    normalize_embeddings=True makes vectors unit-length, so cosine similarity in Chroma
    behaves exactly like a dot product and scores land in a predictable 0..1 range.
    The first call downloads the model (~130 MB for bge-small) into the HuggingFace cache.
    """
    # Imported lazily: torch is heavy, and tests use fake embeddings instead.
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model %s on %s", model_name, device)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
