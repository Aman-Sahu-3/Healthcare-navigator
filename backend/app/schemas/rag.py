"""Response models for RAG answers — reused by the API in Phase 4."""

from typing import Literal

from pydantic import BaseModel, Field

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


class Citation(BaseModel):
    id: int = Field(description="Marker number used in the answer text, e.g. 1 for [1]")
    source: str = Field(description="PDF file name")
    title: str
    page: int = Field(ge=1)
    snippet: str = Field(description="Start of the retrieved passage")
    similarity: float = Field(ge=-1.0, le=1.0)


class Confidence(BaseModel):
    level: ConfidenceLevel
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str]


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: Confidence
    insufficient_evidence: bool
    retrieved_chunks: int
    model: str
    latency_ms: int
