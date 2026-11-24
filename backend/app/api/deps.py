"""Shared services, created once at startup and injected into routes."""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field

from fastapi import Depends, Header, Request
from langchain_chroma import Chroma

from app.api.errors import APIError
from app.core.config import Settings, get_settings
from app.rag.chain import RAGService

logger = logging.getLogger(__name__)


SOURCES_CACHE_TTL_S = 600.0


@dataclass
class Services:
    store: Chroma | None
    rag: RAGService | None
    llm_error: str | None = None
    ingest_lock: threading.Lock = field(default_factory=threading.Lock)
    # Listing documents reads every chunk's metadata, which is slow on a large index.
    # Cache it; ingestion invalidates it, and a TTL covers out-of-band CLI ingests.
    _sources: list[dict] | None = field(default=None, repr=False)
    _sources_at: float = field(default=0.0, repr=False)
    _sources_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def list_sources(self) -> list[dict]:
        from app.ingestion.vectorstore import list_sources

        if self.store is None:
            return []
        with self._sources_lock:
            fresh = time.monotonic() - self._sources_at < SOURCES_CACHE_TTL_S
            if self._sources is None or not fresh:
                self._sources = list_sources(self.store)
                self._sources_at = time.monotonic()
            return self._sources

    def invalidate_sources(self) -> None:
        with self._sources_lock:
            self._sources = None


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


def start_auto_ingest(services: Services, settings: Settings) -> threading.Thread | None:
    """Ingest RAW_DOCS_DIR in a background thread if the index is empty.

    Runs in the background so the API starts (and passes health checks) immediately;
    /health reports "index is empty" until it finishes.
    """
    from app.ingestion.pipeline import discover_pdfs, ingest_path
    from app.ingestion.vectorstore import count

    store = services.store
    if store is None or count(store) > 0 or not discover_pdfs(settings.raw_docs_dir):
        return None

    def _run() -> None:
        if not services.ingest_lock.acquire(blocking=False):
            return
        try:
            logger.info("Auto-ingest: indexing %s", settings.raw_docs_dir)
            report = ingest_path(settings.raw_docs_dir, settings, store=store)
            logger.info("Auto-ingest finished: %s", report.summary())
        except Exception:
            logger.exception("Auto-ingest failed")
        finally:
            services.invalidate_sources()
            services.ingest_lock.release()

    thread = threading.Thread(target=_run, name="auto-ingest", daemon=True)
    thread.start()
    return thread


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
