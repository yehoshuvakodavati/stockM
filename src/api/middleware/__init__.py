"""
StockM v1.0 - Phase 9, Lesson 9
Middleware
==========

Cross-cutting concerns applied to EVERY request: a unique request ID, request
timing, structured access logging, CORS, and security headers. Middleware runs
OUTSIDE routes — routes never need to know about request IDs or CORS.

Execution flow (FastAPI/Starlette):
    request -> [RequestID -> Timing/Logging -> CORS -> SecurityHeaders] -> route
    response <- [route -> SecurityHeaders <- CORS <- Timing/Logging <- RequestID] <- response

Each middleware wraps the next, so the request is enriched as it flows inward
and the response is enriched as it flows outward. Order matters: RequestID
first (so later middleware can log with it), CORS early (browsers need it).
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import Settings
from api.routes.metrics import APIMetrics

logger = logging.getLogger("stockm.api.access")


def register_middleware(app: FastAPI, settings: Settings) -> None:
    """Attach all middleware to the app (called from main.create_app)."""
    # In-process metrics counter shared by middleware + /metrics endpoint.
    app.state.metrics = APIMetrics()

    # 1) Request ID (outermost so every downstream log/resp can reference it).
    app.add_middleware(RequestIDMiddleware)
    # 2) Timing + access logging.
    app.add_middleware(AccessLogMiddleware)
    # 3) Security headers (X-Content-Type-Options, etc.).
    app.add_middleware(SecurityHeadersMiddleware)
    # 4) CORS (added last so it wraps outermost for browser preflight).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID to every request + response.

    If the client sends one, reuse it (distributed tracing); else generate one.
    Stored on ``request.state.request_id`` so handlers and error responses can
    include it. This is the single most useful middleware for debugging: a user
    quotes the request_id and ops finds the exact log line in one grep.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Log every request (method, path, status, latency) and time it.

    The access log is the audit trail: who called what, when, how long. In
    production this feeds the monitoring dashboard (Lesson 10).
    """

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000
        rid = getattr(request.state, "request_id", "-")
        is_error = response.status_code >= 400
        logger.info(
            "%s %s %d %.1fms rid=%s",
            request.method, request.url.path, response.status_code, latency_ms, rid,
        )
        # Update the metrics counter (if present).
        metrics = getattr(request.app.state, "metrics", None)
        if metrics is not None:
            metrics.record_request(latency_ms, is_error, request.url.path)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add basic security headers to every response.

    These mitigate common web vulnerabilities: X-Content-Type-Options stops
    MIME sniffing, X-Frame-Options stops clickjacking. They're cheap insurance
    for when the API is exposed to browsers.
    """

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
