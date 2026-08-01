"""
StockM v1.0 - Phase 9, Lesson 2
FastAPI Application Composition Root
====================================

Builds the FastAPI ``app`` and owns the APPLICATION LIFESPAN: the startup event
that loads the Model Registry once (expensive — seconds) and the shutdown event
that releases it. Requests then pay only inference cost, never load cost.

Why a composition root
----------------------
This is the ONE place that knows about all the layers and wires them together:
it creates the registry, the prediction service, and registers the routers.
Everywhere else depends on abstractions injected in (Lesson 4/5). Changing the
wiring (e.g. a remote model store) changes only this file.

Lifespan (FastAPI's modern startup/shutdown)
--------------------------------------------
FastAPI's ``lifespan`` context manager replaces the old ``@app.on_event`` hooks.
It runs startup code BEFORE the server accepts requests, and shutdown code after
all in-flight requests finish. We load the Model Registry here so it is ready
for the first request. The registry reference is stored on ``app.state`` — the
shared place FastAPI provides for app-wide singletons.

Lesson 2 scope: the lifecycle + wiring + a health route. Routes for
/models, /predict etc. are added in Lesson 7 (they're imported here once built).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import settings

logger = logging.getLogger("stockm.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup + shutdown lifecycle.

    STARTUP: load the Model Registry (every deployed model's metadata, and the
    active model per symbol). This is the expensive step — done ONCE, not per
    request. Stored on ``app.state.registry`` so dependency injection can hand
    it to routes without reloading.

    SHUTDOWN: release resources. For v1 the registry holds joblib/torch objects
    that Python GC reclaims; explicit teardown is a hook for future connection
    pools / GPU contexts.
    """
    # Lazy import: the registry imports torch/joblib; keep it out of the module
    # top level so importing `api.main` (e.g. for tests) doesn't drag the heavy
    # ML stack in unless the server actually starts.
    from api.model_registry import ModelRegistry

    logger.info("API starting — loading model registry...")
    registry = ModelRegistry()
    registry.load_all()  # discover + load all deployed models (Lesson 4)
    app.state.registry = registry
    logger.info(
        "model registry ready: %d symbols loaded", registry.count_symbols(),
    )

    yield  # --- server accepts requests here ---

    # --- shutdown ---
    logger.info("API shutting down — releasing model registry.")
    app.state.registry = None


def create_app() -> FastAPI:
    """Application factory: build + wire the FastAPI app.

    A factory (rather than a module-level ``app = FastAPI()``) is the
    production pattern: tests can build an isolated app per test, and future
    multi-worker / multi-app setups compose cleanly. The factory is the single
    wiring point — add routers and middleware here.
    """
    app = FastAPI(
        title=settings.api_title,
        description=(
            "StockM Prediction API — serves AI predictions from trained ML/DL "
            "models. Submit a symbol to get a BUY/HOLD/SELL signal with the "
            "predicted return and model provenance."
        ),
        version=settings.api_version,
        lifespan=lifespan,  # startup loads the registry; shutdown releases it
    )

    # --- Routers (Lesson 7 fills these in; Lesson 2 ships health inline) ---
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    """Attach all route groups. Imported lazily so a missing router doesn't
    break app creation during early development."""
    from api.routes.health import router as health_router

    app.include_router(health_router)
    # Lesson 7 adds: models, predict, metrics routers.


# Module-level app for `uvicorn api.main:app`. Built via the factory so the
# lifespan + wiring run identically in dev, test, and prod.
app = create_app()
