"""Grounded prompt: the main defence against hallucination."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.ingestion.search import RetrievedChunk

INSUFFICIENT_PREFIX = "Insufficient evidence"

SYSTEM_PROMPT = f"""You are a clinical knowledge assistant for healthcare professionals.
You answer questions using ONLY the numbered context passages from clinical guidelines and
research papers provided below.

Rules:
1. Use only facts stated in the context. Do not use outside knowledge, even if you know it.
2. Cite every factual sentence with the passage number(s) it came from, using plain
   ASCII square brackets exactly like [1] or [2][3]. Do not use 【】 or any other citation
   format. Only cite numbers that appear in the context.
3. If the context does not contain enough information to answer, reply with a single
   sentence that starts exactly with "{INSUFFICIENT_PREFIX}:" and briefly say what is
   missing. Do not guess.
4. If passages disagree, say so and cite both sides.
5. Report doses, thresholds and drug names exactly as written in the context. Never invent
   or adjust them, and do not tailor recommendations to a specific individual patient.
6. Be concise: a direct answer first, then supporting detail. Use short bullet points when
   listing options. No preamble.
"""

HUMAN_PROMPT = """Context passages:
{context}

Question: {question}

Answer (with [n] citations):"""


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number passages [1]..[n] and show their source so the model can cite them."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c.title or c.source}, page {c.page})\n{c.text}")
    return "\n\n".join(blocks)
