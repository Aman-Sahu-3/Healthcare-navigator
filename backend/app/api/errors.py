"""One error format for every failure:

{"error": {"code": "...", "message": "...", "details": [...]}, "request_id": "..."}
"""

from __future__ import annotations

import logging

import openai
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.rag.llm import LLMNotConfiguredError

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: list[dict] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details, self.headers = details, headers


def error_response(
    request: Request,
    status: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    body["request_id"] = getattr(request.state, "request_id", None)
    return JSONResponse(body, status_code=status, headers=headers)


def _clean_validation_errors(exc: RequestValidationError) -> list[dict]:
    out = []
    for e in exc.errors():
        loc = [str(p) for p in e.get("loc", []) if p != "body"]
        out.append({"field": ".".join(loc) or "body", "message": e.get("msg", "invalid")})
    return out


# (exception type, status, code, message) — upstream LLM failures, most specific first
_UPSTREAM: list[tuple[type[Exception], int, str, str]] = [
    (openai.AuthenticationError, 502, "llm_auth_error", "The LLM provider rejected the API key"),
    (openai.RateLimitError, 503, "llm_rate_limited", "LLM provider rate limit or quota reached"),
    (openai.APITimeoutError, 504, "llm_timeout", "The LLM provider timed out"),
    (openai.APIConnectionError, 502, "llm_unreachable", "Could not reach the LLM provider"),
    (openai.APIError, 502, "llm_error", "The LLM provider returned an error"),
]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        return error_response(request, exc.status, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return error_response(
            request, 422, "validation_error", "Invalid request", _clean_validation_errors(exc)
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return error_response(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(LLMNotConfiguredError)
    async def _llm_config(request: Request, exc: LLMNotConfiguredError):
        return error_response(request, 503, "llm_not_configured", str(exc))

    @app.exception_handler(openai.APIError)
    async def _upstream(request: Request, exc: openai.APIError):
        for exc_type, status, code, message in _UPSTREAM:
            if isinstance(exc, exc_type):
                logger.warning("LLM upstream error (%s): %s", code, exc)
                headers = {"Retry-After": "30"} if code == "llm_rate_limited" else None
                return error_response(request, status, code, message, headers=headers)
        return error_response(request, 502, "llm_error", "The LLM provider returned an error")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return error_response(request, 500, "internal_error", "Internal server error")
