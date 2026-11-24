"""FastAPI application factory.

uvicorn app.main:app --reload --reload-dir app --port 8000
open http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import health, ingest, query, sources
from app.api.deps import Services, build_services, start_auto_ingest
from app.api.errors import register_error_handlers
from app.api.middleware import register_middleware
from app.api.ratelimit import RateLimiter, parse_rate
from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
Evidence-based answers from clinical guidelines and research papers, with page-level
citations and a HIGH / MEDIUM / LOW confidence score.

**For healthcare professionals. Not a substitute for clinical judgment.**
"""


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app. Pass `services` to inject test doubles; otherwise they load at startup."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.environment != "local")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "services", None) is None:
            logger.info("Loading embedding model, vector store and LLM...")
            app.state.services = build_services(settings)
            if settings.auto_ingest_on_startup:
                start_auto_ingest(app.state.services, settings)
            logger.info("Startup complete")
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
    )
    app.state.services = services
    app.state.rate_limiter = RateLimiter(*parse_rate(settings.rate_limit))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    register_middleware(app)
    register_error_handlers(app)

    for module in (health, query, sources, ingest):
        app.include_router(module.router)
    return app


app = create_app()
