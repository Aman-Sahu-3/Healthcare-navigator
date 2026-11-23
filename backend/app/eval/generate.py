"""Bootstrap a golden set: ask the LLM to write one question per sampled chunk.

Generated questions are a DRAFT. Review them by hand (delete vague ones, fix answers)
before trusting the numbers — evaluating a model on questions it wrote is optimistic.
"""

from __future__ import annotations

import logging
import random
import re
import time

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.eval.dataset import GoldenItem
from app.eval.llm_json import parse_json_object

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 400

GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write evaluation questions for a clinical guideline search assistant. "
            "Given one passage, write ONE specific question a clinician might ask that is "
            "fully answered by that passage alone, plus a 1-2 sentence answer that uses only "
            "the passage. Do not mention 'the passage' or 'the document' in the question. "
            "If the passage is a reference list, table of contents or has no clinical "
            'content, return {{"skip": true}}. '
            'Return ONLY JSON: {{"question": "...", "answer": "..."}}',
        ),
        ("human", "Passage (from {source}, page {page}):\n{text}"),
    ]
)

# Clearly out-of-scope questions: the assistant must refuse these.
# Add your own "near-miss" negatives too (medical topics NOT covered by your PDFs) —
# those are the hardest and most informative.
NEGATIVE_QUESTIONS = [
    "What is the capital of France?",
    "Who won the 2018 FIFA World Cup?",
    "How do I bake sourdough bread at home?",
    "What is the best laptop for programming?",
    "Explain how a blockchain reaches consensus.",
    "What is the recommended treatment for snake bite envenomation in dogs?",
    "What is the boiling point of mercury?",
    "Summarise the plot of Hamlet.",
]

_REFERENCE_LIKE = re.compile(r"(doi:|et al\.|https?://|\bvol\.\s*\d)", re.IGNORECASE)


def _looks_like_references(text: str) -> bool:
    return len(_REFERENCE_LIKE.findall(text)) >= 3


def sample_chunks(
    store: Chroma, n: int, seed: int = 42, min_chars: int = MIN_CHUNK_CHARS
) -> list[tuple[str, dict]]:
    """Pick up to n substantial chunks, spread evenly across source documents."""
    data = store.get(include=["documents", "metadatas"])
    by_source: dict[str, list[tuple[str, dict]]] = {}
    for text, meta in zip(data["documents"], data["metadatas"], strict=True):
        if len(text) >= min_chars and not _looks_like_references(text):
            by_source.setdefault(meta["source"], []).append((text, meta))
    rng = random.Random(seed)
    for chunks in by_source.values():
        rng.shuffle(chunks)
    picked: list[tuple[str, dict]] = []
    sources = sorted(by_source)
    while len(picked) < n and any(by_source[s] for s in sources):
        for s in sources:
            if by_source[s] and len(picked) < n:
                picked.append(by_source[s].pop())
    return picked


def generate_golden(
    store: Chroma,
    llm: BaseChatModel,
    n: int = 30,
    include_negatives: bool = True,
    sleep_s: float = 0.0,
    seed: int = 42,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[GoldenItem]:
    chain = GEN_PROMPT | llm
    items: list[GoldenItem] = []
    for text, meta in sample_chunks(store, n, seed=seed, min_chars=min_chars):
        reply = chain.invoke({"source": meta["source"], "page": meta["page"], "text": text})
        data = parse_json_object(getattr(reply, "content", str(reply)))
        if not data or data.get("skip") or not data.get("question"):
            logger.info("skipped chunk %s p.%s", meta["source"], meta["page"])
            continue
        items.append(
            GoldenItem(
                id=f"gen{len(items) + 1:03d}",
                question=str(data["question"]).strip(),
                answerable=True,
                expected_sources=[meta["source"]],
                expected_pages=[int(meta["page"])],
                reference_answer=str(data.get("answer", "")).strip() or None,
                origin="generated",
            )
        )
        if sleep_s:
            time.sleep(sleep_s)

    if include_negatives:
        for i, q in enumerate(NEGATIVE_QUESTIONS, 1):
            items.append(
                GoldenItem(id=f"neg{i:02d}", question=q, answerable=False, origin="negative")
            )
    return items
