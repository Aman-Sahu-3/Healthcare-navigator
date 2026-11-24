"""Shared services, created once at startup and injected into routes."""

from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass, field

from fastapi import Depends, Header, Request
from langchain_chroma import Chroma

from app.api.errors import APIError
from app.core.config import Settings, get_settings
from app.rag.chain import RAGService

logger = logging.getLogger(__name__)


@dataclass
class Services:
    store: Chroma | None
    rag: RAGService | None
    llm_error: str | None = None
    ingest_lock: threading.Lock = field(default_factory=threading.Lock)


def build_services(settings: Settings) -> Services:
    """Load the embedding model + vector store once, and the LLM if configured."""
    from app.ingestion.pipeline import build_store
    from app.rag.llm import LLMNotConfiguredError, get_llm

    store = build_store(settings)
    try:
        rag = RAGService(store, get_llm(settings), settings)
        llm_error = None
    except LLMNotConfiguredError as exc:
        logger.warning("LLM not configured: %s", exc)
        rag, llm_error = None, str(exc)
    return Services(store=store, rag=rag, llm_error=llm_error)


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_store(services: Services = Depends(get_services)) -> Chroma:
    if services.store is None:
        raise APIError(503, "index_unavailable", "Vector store is not available")
    return services.store


def get_rag(services: Services = Depends(get_services)) -> RAGService:
    if services.rag is None:
        raise APIError(
            503,
            "llm_not_configured",
            services.llm_error or "The language model is not configured on the server",
        )
    return services.rag


def require_admin(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.admin_api_key is None:
        raise APIError(503, "ingestion_disabled", "Set ADMIN_API_KEY to enable ingestion")
    expected = settings.admin_api_key.get_secret_value()
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise APIError(401, "unauthorized", "Missing or invalid X-API-Key header")
