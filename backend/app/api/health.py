from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import Services, get_services
from app.core.config import Settings, get_settings
from app.ingestion.vectorstore import count
from app.schemas.api import HealthResponse, IndexStats

router = APIRouter(tags=["meta"])


@router.get("/health/live", summary="Liveness probe (cheap, no dependencies)")
def live() -> dict[str, str]:
    """For Docker / platform health checks: answers instantly without touching the index."""
    return {"status": "alive"}


@router.get("/health", response_model=HealthResponse, summary="Readiness and status details")
def health(
    settings: Settings = Depends(get_settings),
    services: Services = Depends(get_services),
) -> HealthResponse:
    issues: list[str] = []
    index = None
    if services.store is None:
        issues.append("vector store unavailable")
    else:
        try:
            index = IndexStats(documents=len(services.list_sources()), chunks=count(services.store))
            if index.chunks == 0:
                status = services.index_status or ""
                if status.startswith("failed"):
                    issues.append(f"index restore from INDEX_URL {status}")
                elif status == "not configured":
                    issues.append("index is empty: ingest documents or set INDEX_URL")
                else:
                    issues.append("index is empty: ingest documents first")
        except Exception:  # health must never crash
            issues.append("vector store not readable")
    if services.rag is None:
        issues.append("LLM not configured")

    return HealthResponse(
        status="degraded" if issues else "ok",
        version=__version__,
        environment=settings.environment,
        llm_model=settings.llm_model,
        llm_configured=services.rag is not None,
        index=index,
        index_source=services.index_status,
        issues=issues,
    )


@router.get("/stats", summary="Usage statistics (aggregate, no question text)")
def stats(services: Services = Depends(get_services)) -> dict:
    return services.stats.snapshot(cache_size=len(services.answer_cache))
