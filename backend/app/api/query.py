import logging

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_rag
from app.api.ratelimit import rate_limit
from app.rag.chain import RAGService
from app.schemas.api import ErrorResponse, QueryRequest, QueryResponse

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)

_ERRORS = {
    422: {"model": ErrorResponse, "description": "Invalid request"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    502: {"model": ErrorResponse, "description": "LLM provider error"},
    503: {"model": ErrorResponse, "description": "LLM not configured or rate limited upstream"},
    504: {"model": ErrorResponse, "description": "LLM provider timed out"},
}


@router.post(
    "/query",
    response_model=QueryResponse,
    responses=_ERRORS,
    dependencies=[Depends(rate_limit)],
    summary="Ask a clinical question",
    description=(
        "Retrieves the most relevant passages from the indexed guidelines, answers using only "
        "those passages, and returns citations (source + page) with a HIGH/MEDIUM/LOW "
        "confidence score. Not a substitute for clinical judgment."
    ),
)
def query(
    body: QueryRequest, request: Request, rag: RAGService = Depends(get_rag)
) -> QueryResponse:
    # Sync endpoint: FastAPI runs it in a worker thread, so the blocking LLM call
    # doesn't stall the event loop.
    result = rag.answer(body.question, k=body.top_k)
    # Log metadata only — never the question text (it may contain patient details).
    logger.info(
        "query answered: confidence=%s citations=%d chunks=%d latency_ms=%d",
        result.confidence.level,
        len(result.citations),
        result.retrieved_chunks,
        result.latency_ms,
        extra={"request_id": getattr(request.state, "request_id", None)},
    )
    return QueryResponse(**result.model_dump())
