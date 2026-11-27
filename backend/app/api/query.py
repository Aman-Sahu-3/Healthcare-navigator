import logging
import time

from fastapi import APIRouter, Depends, Request

from app.api.cache import cache_key
from app.api.deps import Services, get_rag, get_services
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
    body: QueryRequest,
    request: Request,
    rag: RAGService = Depends(get_rag),
    services: Services = Depends(get_services),
) -> QueryResponse:
    # Sync endpoint: FastAPI runs it in a worker thread, so the blocking LLM call
    # doesn't stall the event loop.
    k = body.top_k or rag.settings.retrieval_top_k
    key = cache_key(body.question, k, rag.model_name)
    start = time.perf_counter()

    hit = services.answer_cache.get(key)
    if hit is not None:
        result = hit.model_copy(
            update={
                "cached": True,
                "question": body.question,  # echo this user's wording, not the first asker's
                "latency_ms": int((time.perf_counter() - start) * 1000),
            }
        )
    else:
        try:
            result = rag.answer(body.question, k=k)
        except Exception as exc:
            services.stats.record_error(type(exc).__name__)
            raise
        services.answer_cache.put(key, result)

    services.stats.record_answer(
        result.confidence.level, result.insufficient_evidence, result.latency_ms, result.cached
    )
    # Log metadata only — never the question text (it may contain patient details).
    logger.info(
        "query answered: confidence=%s citations=%d chunks=%d latency_ms=%d cached=%s",
        result.confidence.level,
        len(result.citations),
        result.retrieved_chunks,
        result.latency_ms,
        result.cached,
        extra={"request_id": getattr(request.state, "request_id", None)},
    )
    return QueryResponse(**result.model_dump())
