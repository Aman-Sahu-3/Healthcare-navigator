from fastapi import APIRouter, Depends

from app.api.deps import Services, get_services, get_store
from app.schemas.api import SourceDocument, SourcesResponse

router = APIRouter(tags=["documents"])


@router.get("/sources", response_model=SourcesResponse, summary="List indexed documents")
def sources(services: Services = Depends(get_services)) -> SourcesResponse:
    get_store(services)  # 503 if the vector store is unavailable
    docs = [SourceDocument(**d) for d in services.list_sources()]
    return SourcesResponse(count=len(docs), documents=docs)
