"""
StockM v1.0 - Phase 9
Dependency Injection
=====================

FastAPI's dependency system wires startup-loaded singletons (registry,
prediction service) into routes WITHOUT the routes importing them at module
level. Tests override these deps with fakes — no real models loaded.
"""
from __future__ import annotations

from fastapi import Depends, Request

from api.model_registry import ModelRegistry
from api.services.prediction_service import PredictionService


def get_registry(request: Request) -> ModelRegistry:
    """Return the startup-loaded ModelRegistry from app state."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise RuntimeError("model registry not loaded — lifespan did not run")
    return registry


def get_prediction_service(request: Request) -> PredictionService:
    """Return the PredictionService (built from the registry at startup)."""
    svc = getattr(request.app.state, "prediction_service", None)
    if svc is None:
        raise RuntimeError("prediction service not configured")
    return svc
