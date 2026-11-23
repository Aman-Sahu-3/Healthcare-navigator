"""LLM-as-judge: is the answer supported by the passages, and does it match the reference?

One call per question grades both, to stay inside free-tier rate limits.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.eval.llm_json import parse_json_object
from app.ingestion.search import RetrievedChunk
from app.rag.prompt import format_context

Grade = Literal["yes", "partial", "no"]
GRADE_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator of a medical question-answering system.\n"
            "faithfulness: is EVERY factual claim in the ANSWER supported by the PASSAGES? "
            "yes = all supported; partial = some unsupported detail; no = key claims "
            "unsupported or contradicted.\n"
            "correctness: does the ANSWER convey the same key facts as the REFERENCE? "
            "yes / partial / no, or n/a if no reference is given.\n"
            'Return ONLY JSON: {{"faithfulness": "yes|partial|no", '
            '"correctness": "yes|partial|no|n/a", "reason": "<one sentence>"}}',
        ),
        (
            "human",
            "PASSAGES:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}\n\n"
            "REFERENCE:\n{reference}",
        ),
    ]
)


class Judgement(BaseModel):
    faithfulness: Grade | None = None
    correctness: Grade | None = None
    reason: str = ""


def _grade(value: object) -> Grade | None:
    v = str(value or "").strip().lower()
    return v if v in GRADE_SCORE else None  # type: ignore[return-value]


def judge_answer(
    llm: BaseChatModel,
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    reference: str | None,
) -> Judgement:
    reply = (JUDGE_PROMPT | llm).invoke(
        {
            "context": format_context(chunks) or "(none)",
            "question": question,
            "answer": answer,
            "reference": reference or "(none)",
        }
    )
    data = parse_json_object(getattr(reply, "content", str(reply)))
    if not data:
        return Judgement(reason="judge returned unparseable output")
    return Judgement(
        faithfulness=_grade(data.get("faithfulness")),
        correctness=_grade(data.get("correctness")) if reference else None,
        reason=str(data.get("reason", ""))[:300],
    )
