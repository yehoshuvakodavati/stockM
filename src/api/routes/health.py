"""Health & info routes — root, health (readiness), version."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.config import get_settings
from api.dependencies import get_registry
from api.model_registry import ModelRegistry
from api.schemas.predict import HealthResponse

router = APIRouter(tags=["info"])


@router.get("/", summary="API root")
def root() -> dict:
    """API root — confirms the service is reachable."""
    s = get_settings()
    return {"service": s.api_title, "status": "ok", "version": s.api_version}


@router.get("/health", response_model=HealthResponse, summary="Readiness check")
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    """Readiness probe: 200 if the process is alive AND models are loaded.

    A liveness probe (process alive) is necessary but not sufficient — a
    process with zero loaded models can't serve predictions. This readiness
    check verifies the registry loaded, so orchestrators only route traffic
    to instances that can actually predict.
    """
    s = get_settings()
    return HealthResponse(
        status="healthy" if registry.is_ready() else "degraded",
        environment=s.environment,
        models_loaded=registry.count_symbols(),
    )


@router.get("/version", summary="Deployed version")
def version() -> dict:
    """Deployed API version (ties to Phase 8 Lesson 13 reproducibility)."""
    s = get_settings()
    return {"api_version": s.api_version, "environment": s.environment}
