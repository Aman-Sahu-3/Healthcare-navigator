"""Request ID + access log for every request."""

from __future__ import annotations

import logging
import re
import time
import uuid

from fastapi import FastAPI, Request

logger = logging.getLogger("app.access")

_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# Polled every few seconds by Docker / the hosting platform: not worth a log line.
_QUIET_PATHS = frozenset({"/health/live"})


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-ID"] = request_id
        if request.url.path in _QUIET_PATHS and response.status_code < 400:
            return response
        logger.info(
            "%s %s -> %s (%d ms)",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response
