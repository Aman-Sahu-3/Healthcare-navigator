from fastapi import APIRouter, Depends
from langchain_chroma import Chroma

from app.api.deps import get_store
from app.ingestion.vectorstore import list_sources
from app.schemas.api import SourceDocument, SourcesResponse

router = APIRouter(tags=["documents"])


@router.get("/sources", response_model=SourcesResponse, summary="List indexed documents")
def sources(store: Chroma = Depends(get_store)) -> SourcesResponse:
    docs = [SourceDocument(**d) for d in list_sources(store)]
    return SourcesResponse(count=len(docs), documents=docs)
