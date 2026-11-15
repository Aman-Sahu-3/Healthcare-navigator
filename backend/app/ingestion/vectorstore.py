"""ChromaDB access: persistent collection configured for cosine similarity."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_vectorstore(
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str,
) -> Chroma:
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def has_file_hash(store: Chroma, file_hash: str) -> bool:
    """True if any chunk from this exact file version is already stored."""
    res = store.get(where={"file_hash": file_hash}, limit=1, include=[])
    return bool(res["ids"])


def delete_source(store: Chroma, source: str) -> int:
    """Remove every chunk for a file name (used when a file's content changed)."""
    res = store.get(where={"source": source}, include=[])
    ids = res["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


def list_sources(store: Chroma) -> list[dict]:
    """Summarize stored documents: one entry per source file."""
    res = store.get(include=["metadatas"])
    summary: dict[str, dict] = {}
    for meta in res["metadatas"]:
        src = meta["source"]
        entry = summary.setdefault(
            src,
            {
                "source": src,
                "title": meta.get("title", src),
                "total_pages": meta.get("total_pages"),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
    return sorted(summary.values(), key=lambda e: e["source"])


def count(store: Chroma) -> int:
    return store._collection.count()
