"""Turn the LLM's [n] markers into citations using retrieval metadata.

The LLM only ever writes numbers. Source names and page numbers come from ChromaDB
metadata, so they can't be hallucinated. Markers pointing at passages that don't exist
are removed from the answer.
"""

from __future__ import annotations

import re

from app.ingestion.search import RetrievedChunk
from app.rag.prompt import INSUFFICIENT_PREFIX
from app.schemas.rag import Citation

# Matches [1], [2, 3], [1-3], [1–3]
_MARKER = re.compile(r"\[(\d+(?:\s*(?:,|-|–)\s*\d+)*)\]")
_SNIPPET_CHARS = 280


def _expand(group: str) -> list[int]:
    nums: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[-–]", part):
            a, b = (int(x) for x in re.split(r"\s*[-–]\s*", part))
            if a <= b and b - a < 50:
                nums.extend(range(a, b + 1))
        else:
            nums.append(int(part))
    return nums


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Return (cleaned answer, citations in order of first mention)."""
    n = len(chunks)
    order: list[int] = []

    def _replace(m: re.Match) -> str:
        valid = [i for i in _expand(m.group(1)) if 1 <= i <= n]
        for i in valid:
            if i not in order:
                order.append(i)
        return "".join(f"[{i}]" for i in dict.fromkeys(valid))

    cleaned = _MARKER.sub(_replace, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)  # tidy spaces left by removed markers
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    citations = []
    for i in order:
        c = chunks[i - 1]
        snippet = c.text if len(c.text) <= _SNIPPET_CHARS else c.text[:_SNIPPET_CHARS] + "..."
        citations.append(
            Citation(
                id=i,
                source=c.source,
                title=c.title or c.source,
                page=max(c.page, 1),
                snippet=snippet,
                similarity=c.similarity,
            )
        )
    return cleaned, citations


def is_insufficient(answer: str) -> bool:
    return answer.strip().lower().startswith(INSUFFICIENT_PREFIX.lower())
