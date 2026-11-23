"""The RAG chain, built with LCEL.

{question, k}
  -> retrieve top-k chunks (cosine, filtered by MIN_SIMILARITY)
  -> no chunks?  -> "Insufficient evidence" (no LLM call, no cost)
  -> number passages [1..n] -> grounded prompt -> LLM -> text
  -> map [n] markers to source/page from metadata -> confidence score
  -> RAGAnswer
"""

from __future__ import annotations

import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from app.core.config import Settings
from app.ingestion.search import retrieve
from app.rag.citations import extract_citations, is_insufficient
from app.rag.confidence import score_confidence
from app.rag.prompt import INSUFFICIENT_PREFIX, build_prompt, format_context
from app.schemas.rag import RAGAnswer

NO_EVIDENCE_ANSWER = (
    f"{INSUFFICIENT_PREFIX}: no passages in the indexed documents are relevant enough "
    "to answer this question."
)


def build_rag_chain(store: Chroma, llm: BaseChatModel, settings: Settings) -> Runnable:
    def _retrieve(x: dict[str, Any]):
        return retrieve(
            store,
            x["question"],
            k=x.get("k") or settings.retrieval_top_k,
            min_similarity=settings.min_similarity,
        )

    generate = RunnablePassthrough.assign(
        context=lambda x: format_context(x["chunks"])
    ) | RunnablePassthrough.assign(raw_answer=build_prompt() | llm | StrOutputParser())
    skip_llm = RunnablePassthrough.assign(raw_answer=lambda _: NO_EVIDENCE_ANSWER)

    def _finalize(x: dict[str, Any]) -> dict[str, Any]:
        chunks = x["chunks"]
        insufficient = is_insufficient(x["raw_answer"])
        answer, citations = extract_citations(x["raw_answer"], chunks)
        if insufficient:
            citations = []
        confidence = score_confidence(
            citations,
            insufficient,
            settings.high_conf_threshold,
            settings.medium_conf_threshold,
        )
        return {
            "question": x["question"],
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "insufficient_evidence": insufficient,
            "retrieved_chunks": len(chunks),
        }

    return (
        RunnablePassthrough.assign(chunks=RunnableLambda(_retrieve))
        | RunnableBranch((lambda x: not x["chunks"], skip_llm), generate)
        | RunnableLambda(_finalize)
    ).with_config(run_name="healthcare_rag")


class RAGService:
    """Holds the vector store + chain; build once at startup, call answer() per request."""

    def __init__(self, store: Chroma, llm: BaseChatModel, settings: Settings):
        self.settings = settings
        self.store = store
        self.model_name = getattr(llm, "model_name", None) or settings.llm_model
        self.chain = build_rag_chain(store, llm, settings)

    def answer(self, question: str, k: int | None = None) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        start = time.perf_counter()
        result = self.chain.invoke({"question": question, "k": k})
        latency = int((time.perf_counter() - start) * 1000)
        return RAGAnswer(**result, model=self.model_name, latency_ms=latency)

    @classmethod
    def from_settings(cls, settings: Settings) -> RAGService:
        from app.ingestion.pipeline import build_store
        from app.rag.llm import get_llm

        return cls(build_store(settings), get_llm(settings), settings)
