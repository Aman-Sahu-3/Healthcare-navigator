"""Request/response models for the REST API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.rag import RAGAnswer


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        examples=["What is the first-line treatment for hypertension in adults?"],
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Passages to retrieve")

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return v


class QueryResponse(RAGAnswer):
    """Answer with citations and confidence. Same shape as the RAG chain output."""


class IndexStats(BaseModel):
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    llm_model: str
    llm_configured: bool
    index: IndexStats | None
    index_source: str | None = Field(
        default=None, description='"local", "restored", "not configured" or "failed: ..."'
    )
    issues: list[str] = Field(default_factory=list)


class SourceDocument(BaseModel):
    source: str
    title: str
    total_pages: int | None
    chunks: int


class SourcesResponse(BaseModel):
    count: int
    documents: list[SourceDocument]


class IngestFileResult(BaseModel):
    source: str
    status: Literal["ingested", "skipped", "replaced", "failed", "empty"]
    pages: int = 0
    chunks: int = 0
    error: str | None = None


class IngestResponse(BaseModel):
    files: list[IngestFileResult]
    new_chunks: int
    seconds: float
    summary: str


class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="Re-embed files even if unchanged")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: str | None = None
