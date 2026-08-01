"""
StockM v1.0 - Phase 9, Lesson 2
Dependency Injection
=====================

FastAPI's dependency system: a function decorated/used with ``Depends`` runs
before the handler, and its return value is injected as a parameter. This is
how the Model Registry (a startup-loaded singleton) and the Prediction Service
reach the routes WITHOUT the routes importing them at module level.

Why dependency injection
------------------------
A route that does ``registry = ModelRegistry()`` inside the handler reloads the
registry every request (seconds of latency). A route that imports a module
global is hard to test (you can't swap the registry for a fake). DI solves
both: the registry is built ONCE at startup and stored on ``app.state``; the
dependency reads it from the request's app state and hands it to the handler.
Tests override the dependency with a fake — no real models loaded.

Lesson 2 scope: the registry dependency. The service dependency arrives in
Lesson 5 (it wraps the registry + the prediction engine).
"""
from __future__ import annotations

from fastapi import Depends, Request

from api.model_registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    """Return the startup-loaded ModelRegistry from app state.

    FastAPI caches a dependency's result within a single request, so calling
    this in multiple handlers (or multiple times in one) reuses one object.

    Raises:
        RuntimeError: if the registry isn't on app.state (lifespan didn't run).
                      The Lesson 8 error handler turns this into a 503.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise RuntimeError("model registry not loaded — lifespan did not run")
    return registry


# Forward-declared service dependency (Lesson 5 implements the service).
# Shown here so the DI pattern is visible in one place; routes use it in L7.
def get_prediction_service(request: Request):
    """Return the PredictionService (built from the registry at startup).

    Lesson 5 builds ``PredictionService`` and stores it on app.state alongside
    the registry. This dep hands it to routes.
    """
    svc = getattr(request.app.state, "prediction_service", None)
    if svc is None:
        # Not yet wired in Lesson 2; will be after Lesson 5.
        raise RuntimeError("prediction service not configured (Lesson 5)")
    return svc
