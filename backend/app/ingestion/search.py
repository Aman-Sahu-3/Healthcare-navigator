"""Raw retrieval (no LLM) — used to sanity-check ingestion and, in Phase 2, by the RAG chain."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    title: str
    chunk_id: str
    similarity: float  # cosine similarity, 0..1 for normalized embeddings


def retrieve(
    store: Chroma, query: str, k: int = 5, min_similarity: float = 0.0
) -> list[RetrievedChunk]:
    """Top-k cosine retrieval. Chroma returns cosine *distance*; similarity = 1 - distance."""
    results = store.similarity_search_with_score(query, k=k)
    chunks: list[RetrievedChunk] = []
    for doc, distance in results:
        sim = round(1.0 - float(distance), 4)
        if sim < min_similarity:
            continue
        m = doc.metadata
        chunks.append(
            RetrievedChunk(
                text=doc.page_content,
                source=m.get("source", "unknown"),
                page=int(m.get("page", 0)),
                title=m.get("title", ""),
                chunk_id=m.get("chunk_id", doc.id or ""),
                similarity=sim,
            )
        )
    return chunks
