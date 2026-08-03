"""
StockM v1.0 - Phase 9
FastAPI Application Composition Root
=====================================

Builds the FastAPI ``app`` and owns the APPLICATION LIFESPAN: startup loads the
Model Registry + Prediction Service once (expensive) and stores them on
``app.state``; shutdown releases them. Requests pay only inference cost.

Composition root: the ONE place that wires layers together — creates the
registry, the prediction service, registers routers, middleware, and error
handlers. Everywhere else depends on abstractions injected in.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import configure_logging, get_settings

logger = logging.getLogger("stockm.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup + shutdown lifecycle."""
    settings = get_settings()
    configure_logging(settings)

    from api.model_registry import ModelRegistry
    from api.services.prediction_service import PredictionService

    logger.info("API starting — env=%s — loading model registry...", settings.environment)
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    app.state.settings = settings
    app.state.prediction_service = PredictionService(
        registry=registry,
        default_split=settings.default_split,
        default_threshold=settings.default_threshold,
    )
    logger.info("model registry ready: %d ML + DL models", len(registry.list_entries()))

    yield  # --- server accepts requests here ---

    logger.info("API shutting down.")
    app.state.registry = None
    app.state.prediction_service = None


def create_app() -> FastAPI:
    """Application factory: build + wire the FastAPI app."""
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        description=(
            "StockM Prediction API — serves AI predictions from trained ML/DL "
            "models. Submit a symbol to get a BUY/HOLD/SELL signal with the "
            "predicted return and model provenance."
        ),
        version=settings.api_version,
        lifespan=lifespan,
    )

    # Middleware (Lesson 9) — added before routes so it wraps all requests.
    from api.middleware import register_middleware
    register_middleware(app, settings)

    # Error handlers (Lesson 8).
    from api.errors import register_exception_handlers
    register_exception_handlers(app)

    # Routers (Lesson 7).
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """Attach all route groups."""
    from api.routes.health import router as health_router
    from api.routes.models import router as models_router
    from api.routes.predict import router as predict_router
    from api.routes.metrics import router as metrics_router

    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(predict_router)
    app.include_router(metrics_router)


app = create_app()
