"""
StockM v1.0 - Phase 9, Lesson 2
Health & Info Routes
=====================

The simplest route group: root, health, version. These don't need the model
registry, so they have no dependencies — useful for liveness probes that
should succeed even if model loading is slow/failing.

``/health`` in Lesson 2 only checks the process. Lesson 4 deepens it to also
report whether the registry loaded (a true readiness check). Keeping the
shallow version now lets us start the server before the registry exists.
"""
from __future__ import annotations

from fastapi import APIRouter

from api.config import get_settings

router = APIRouter(tags=["info"])


@router.get("/")
def root() -> dict:
    """API root — confirms the service is reachable."""
    s = get_settings()
    return {"service": s.api_title, "status": "ok"}


@router.get("/health")
def health() -> dict:
    """Liveness probe. Returns 200 if the process is alive.

    Lesson 4 upgrades this to a readiness check that verifies the registry.
    """
    return {"status": "healthy", "environment": get_settings().environment}


@router.get("/version")
def version() -> dict:
    """Deployed API version (ties to Phase 8 Lesson 13 reproducibility)."""
    return {"api_version": get_settings().api_version, "environment": get_settings().environment}
