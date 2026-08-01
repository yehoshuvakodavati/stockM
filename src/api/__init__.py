"""
StockM v1.0 - Phase 9, Lesson 2
Prediction Service (FastAPI) — package root
==========================================

Exposes StockM's trained ML/DL models over a scalable REST API. This package is
a THIN TRANSPORT LAYER: it owns HTTP contracts (requests/responses, status
codes, routing) and delegates all domain logic to the existing, verified
prediction engine (models.prediction / deep_learning.inference) and backtesting
modules. No prediction logic is duplicated here.

Layered architecture (clean architecture / separation of concerns)
------------------------------------------------------------------
    routes/      HTTP layer    — parse request, call service, format response.
                                 Knows about HTTP; knows NOTHING about models.
    services/    business layer— predict / predict_single_stock / predict_batch.
                                 Knows about predictions; knows NOTHING about HTTP.
    dependencies.py  DI wires  — registry + services injected into routes.
    schemas/     contract layer— Pydantic request/response models (the JSON API).
    middleware.py cross-cutting— logging, timing, CORS, request IDs (Lesson 9).
    config.py    settings layer— env-driven config, dev/test/prod (Lesson 3).
    main.py      composition   — builds the FastAPI app + lifespan + wiring.

The dependency direction is one-way: routes -> services -> prediction engine.
The engine never imports the API. This keeps the prediction code reusable from
scripts, backtests, AND the API without coupling it to HTTP.

Public API
----------
    app: the configured FastAPI application (importable as `api.main:app`).
"""
from __future__ import annotations
