"""
StockM v1.0 - Phase 9, Lesson 8
Centralized Error Handling
===========================

Custom exceptions + FastAPI exception handlers. Every error returns a uniform
JSON envelope (ErrorResponse) with the right HTTP status code. Centralizing
this means a missing model, a bad request, and a crash all surface the same
way to the client — predictable, parseable, no leaks of internal tracebacks.

Status code mapping
-------------------
    ValidationError (Pydantic)   -> 422 Unprocessable Entity (FastAPI default)
    NotFoundError (no model)     -> 404 Not Found
    BadRequestError (bad input)  -> 400 Bad Request
    ModelLoadError / registry    -> 503 Service Unavailable
    ServiceError (unexpected)    -> 500 Internal Server Error

Security: never return a stack trace to the client. The 500 handler logs the
full traceback server-side and returns a generic message + the request_id (so
the user can quote it and ops can find the log line).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas.predict import ErrorResponse

logger = logging.getLogger("stockm.api.errors")


# -------------------------------------------------- custom exceptions

class APIError(Exception):
    """Base class for API errors carrying a status code + error_code."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, detail: str, status_code: int | None = None,
                 error_code: str | None = None) -> None:
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        super().__init__(detail)


class NotFoundError(APIError):
    """A requested resource (e.g. a symbol's model) doesn't exist."""
    status_code = 404
    error_code = "NOT_FOUND"


class BadRequestError(APIError):
    """The request is malformed beyond Pydantic's field validation."""
    status_code = 400
    error_code = "BAD_REQUEST"


class ModelNotAvailableError(APIError):
    """No deployed model for the symbol (registry miss)."""
    status_code = 404
    error_code = "MODEL_NOT_FOUND"


class PredictionError(APIError):
    """The prediction engine raised during inference."""
    status_code = 503
    error_code = "PREDICTION_FAILED"


# -------------------------------------------------- handlers

def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the app (called from main.py)."""

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):
        logger.warning("APIError %s: %s (path=%s)", exc.error_code, exc.detail,
                       request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=exc.detail, error_code=exc.error_code,
                                  request_id=_request_id(request)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        # Pydantic 422 — enrich with the request_id so the client can correlate.
        detail = "; ".join(
            f"{'.'.join(str(e['loc']))}: {e['msg']}" for e in exc.errors()
        )
        logger.info("validation error: %s (path=%s)", detail, request.url.path)
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail=detail, error_code="VALIDATION_ERROR",
                                  request_id=_request_id(request)).model_dump(),
        )

    @app.exception_handler(KeyError)
    async def handle_key_error(request: Request, exc: KeyError):
        # A KeyError from the registry -> 404 model not found.
        logger.warning("KeyError: %s (path=%s)", exc, request.url.path)
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(detail=f"resource not found: {exc}",
                                  error_code="NOT_FOUND",
                                  request_id=_request_id(request)).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        # Catch-all: log the full traceback server-side, return a generic 500.
        logger.exception("unexpected error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="internal server error",
                error_code="INTERNAL_ERROR",
                request_id=_request_id(request),
            ).model_dump(),
        )
