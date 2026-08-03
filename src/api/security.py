"""
StockM v1.0 - Phase 9, Lesson 14
Production Readiness: Security & Rate Limiting
================================================

Authentication (API key) and rate limiting, wired as FastAPI dependencies so
they're opt-in per endpoint (protect /predict, leave /health open). This module
is the production-hardening layer; Lessons 1-13 built the functional API, this
makes it safe to expose.

API key auth
------------
If ``settings.api_key`` is set (via the API_KEY env var), protected endpoints
require ``Authorization: Bearer <key>`` (or ``X-API-Key: <key>``). If the key
is unset, auth is disabled (development). This is the minimal honest auth: a
shared secret. JWT/OAuth2 would live here in a fuller system (the dependency
contract is the same: a callable returning the authenticated identity).

Rate limiting
-------------
A simple in-memory token-bucket per client IP. v1 is single-worker, so in-memory
is fine; for multi-worker, Redis would back this (documented as such). Limits
protect the service from abuse (and from a buggy client hammering /predict).

Scaling notes (Lesson 14 discussion, not all coded here)
-------------------------------------------------------
- Horizontal scaling: run N uvicorn workers (CPU-bound inference) or N
  containers behind a load balancer. The app is stateless (registry is
  read-only after startup) so any instance can serve any request.
- Caching: repeated /predict for the same symbol+date is cacheable (the model
  output is deterministic). A Redis/LRU cache layer would sit in the service.
- HTTPS: terminate TLS at a reverse proxy (nginx/traefik), not in the app.
- Async: /predict is sync (CPU-bound). /predict/batch could be async with
  asyncio.gather over a thread pool for true concurrency (Lesson 14 exercise).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from api.config import Settings, get_settings

logger = logging.getLogger("stockm.api.security")

# Accept the key via either header (Bearer-style or explicit X-API-Key).
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    request: Request,
    provided_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """Dependency: enforce the API key on protected endpoints.

    Returns the authenticated identity (the key) on success. If no key is
    configured, auth is disabled (dev mode). Use as:
        @router.post("/predict", dependencies=[Depends(require_api_key)])
    """
    configured = settings.api_key
    if not configured:
        # No key configured -> auth disabled (development). Log once.
        return "anonymous"
    # Also accept Authorization: Bearer <key> via the request headers.
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        provided_key = provided_key or auth[len("Bearer "):]
    if not provided_key or provided_key != configured:
        raise HTTPException(status_code=401, detail="invalid or missing API key",
                            headers={"WWW-Authenticate": "ApiKey"})
    return provided_key


# ---------------------------------------------------------------------------
# Rate limiter (in-memory token bucket per client IP)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-client token-bucket rate limiter.

    v1: in-memory, single-worker. Each client gets a bucket refilled at
    ``rate`` tokens/sec up to ``capacity``. A request costs 1 token; if the
    bucket is empty, 429 Too Many Requests. For multi-worker, back this with
    Redis (the interface is the same).
    """

    def __init__(self, rate: float = 10.0, capacity: float = 20.0) -> None:
        self.rate = rate            # tokens added per second
        self.capacity = capacity    # max bucket size
        self._buckets: dict[str, tuple[float, float]] = defaultdict(
            lambda: (capacity, time.monotonic())
        )

    def check(self, client_id: str) -> bool:
        """True if the request is allowed; False if rate-limited."""
        tokens, last = self._buckets[client_id]
        now = time.monotonic()
        # Refill since last call.
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._buckets[client_id] = (tokens, now)
            return False
        self._buckets[client_id] = (tokens - 1.0, now)
        return True


# Singleton rate limiter (lives on app.state in production).
_limiter: RateLimiter | None = None


def get_rate_limiter(request: Request) -> RateLimiter:
    """Return the app's rate limiter (created lazily on app.state)."""
    global _limiter
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        limiter = _limiter or RateLimiter()
        request.app.state.rate_limiter = limiter
        _limiter = limiter
    return limiter


def rate_limit(request: Request, limiter: RateLimiter = Depends(get_rate_limiter)) -> None:
    """Dependency: enforce per-IP rate limiting. Returns 429 if exceeded."""
    client = request.client.host if request.client else "unknown"
    if not limiter.check(client):
        raise HTTPException(status_code=429, detail="rate limit exceeded; slow down")
