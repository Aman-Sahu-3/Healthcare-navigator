from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import Services, get_services
from app.core.config import Settings, get_settings
from app.ingestion.vectorstore import count, list_sources
from app.schemas.api import HealthResponse, IndexStats

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
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
            index = IndexStats(
                documents=len(list_sources(services.store)), chunks=count(services.store)
            )
            if index.chunks == 0:
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
        issues=issues,
    )
