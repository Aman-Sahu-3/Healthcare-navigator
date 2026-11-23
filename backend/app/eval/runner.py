"""Run the golden set through retrieval + the full RAG chain and record per-question results."""

from __future__ import annotations

import logging
import time

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from app.eval.dataset import GoldenItem
from app.eval.judge import Judgement, judge_answer
from app.ingestion.search import retrieve
from app.rag.chain import RAGService

logger = logging.getLogger(__name__)


class ItemResult(BaseModel):
    id: str
    question: str
    answerable: bool
    origin: str
    has_expected: bool  # answerable and has expected_sources -> retrieval metrics apply
    # retrieval (computed with no similarity floor, so we see the raw ranking)
    retrieved: list[tuple[str, int, float]]  # (source, page, similarity)
    source_rank: int | None  # 1-based rank of first chunk from an expected source
    page_hit: bool | None
    top_similarity: float
    # generation
    answer: str
    refused: bool
    cited_sources: list[str]
    cited_expected: bool | None
    confidence: str
    confidence_score: float
    latency_ms: int
    judgement: Judgement | None = None
    error: str | None = None

    @property
    def refusal_correct(self) -> bool:
        return self.refused != self.answerable


def _source_rank(item: GoldenItem, retrieved: list[tuple[str, int, float]]) -> int | None:
    for rank, (src, _, _) in enumerate(retrieved, 1):
        if src in item.expected_sources:
            return rank
    return None


def _page_hit(item: GoldenItem, retrieved: list[tuple[str, int, float]]) -> bool | None:
    if not item.expected_pages:
        return None
    return any(
        src in item.expected_sources and page in item.expected_pages for src, page, _ in retrieved
    )


def evaluate_item(
    service: RAGService,
    item: GoldenItem,
    k: int,
    judge_llm: BaseChatModel | None,
) -> ItemResult:
    settings = service.settings
    raw = retrieve(service.store, item.question, k=k, min_similarity=0.0)
    retrieved = [(c.source, c.page, c.similarity) for c in raw]
    has_expected = item.answerable and bool(item.expected_sources)

    base = dict(
        id=item.id,
        question=item.question,
        answerable=item.answerable,
        origin=item.origin,
        has_expected=has_expected,
        retrieved=retrieved,
        source_rank=_source_rank(item, retrieved) if has_expected else None,
        page_hit=_page_hit(item, retrieved) if has_expected else None,
        top_similarity=retrieved[0][2] if retrieved else 0.0,
    )
    try:
        ans = service.answer(item.question, k=k)
    except Exception as exc:  # record and continue; one API hiccup shouldn't kill the run
        logger.warning("%s failed: %s", item.id, exc)
        return ItemResult(
            **base,
            answer="",
            refused=False,
            cited_sources=[],
            cited_expected=None,
            confidence="LOW",
            confidence_score=0.0,
            latency_ms=0,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )

    cited = list(dict.fromkeys(c.source for c in ans.citations))
    judgement = None
    if judge_llm is not None and item.answerable and not ans.insufficient_evidence:
        context = [c for c in raw if c.similarity >= settings.min_similarity]
        try:
            judgement = judge_answer(
                judge_llm, item.question, ans.answer, context, item.reference_answer
            )
        except Exception as exc:
            judgement = Judgement(reason=f"judge error: {type(exc).__name__}")

    return ItemResult(
        **base,
        answer=ans.answer,
        refused=ans.insufficient_evidence,
        cited_sources=cited,
        cited_expected=(
            any(s in item.expected_sources for s in cited)
            if has_expected and not ans.insufficient_evidence
            else None
        ),
        confidence=ans.confidence.level,
        confidence_score=ans.confidence.score,
        latency_ms=ans.latency_ms,
        judgement=judgement,
    )


def run_eval(
    service: RAGService,
    items: list[GoldenItem],
    k: int | None = None,
    judge_llm: BaseChatModel | None = None,
    sleep_s: float = 0.0,
    progress: bool = True,
) -> list[ItemResult]:
    k = k or service.settings.retrieval_top_k
    results = []
    for n, item in enumerate(items, 1):
        r = evaluate_item(service, item, k, judge_llm)
        results.append(r)
        if progress:
            mark = "ERR" if r.error else ("ok " if r.refusal_correct else "BAD")
            print(f"[{n:>3}/{len(items)}] {mark} {r.confidence:<6} {item.id}: {item.question[:60]}")
        if sleep_s:
            time.sleep(sleep_s)
    return results
