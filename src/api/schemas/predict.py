"""
StockM v1.0 - Phase 9, Lesson 6
Request/Response Schemas (Pydantic)
====================================

The JSON API contract. Pydantic models validate incoming requests (wrong types,
missing fields, bad symbol format -> 422 with a clear message) and serialize
outgoing responses (the response shape is frozen, independent of internal dict
changes). Define once, use for both directions.

Why schemas (not raw dicts)
---------------------------
A raw dict response couples the API contract to an internal structure that can
change when the prediction engine evolves. A Pydantic response model freezes
the contract: clients can rely on ``predicted_return`` always being a float.
Request validation is the other half: a missing ``symbol`` or a non-numeric
``threshold`` becomes a 422 (not a 500) with a field-specific error message.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# -------------------------------------------------- common

class ErrorResponse(BaseModel):
    """Standard error envelope (Lesson 8 returns this for all errors)."""
    detail: str
    error_code: str = "ERROR"
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    environment: str
    models_loaded: int


# -------------------------------------------------- predict

class PredictRequest(BaseModel):
    """Request body for POST /predict.

    Only ``symbol`` is required; everything else has a sensible default so the
    common case (latest prediction for a symbol) is a one-field body.
    """
    symbol: str = Field(..., min_length=1, max_length=20,
                        description="Ticker, e.g. 'RELIANCE.NS'",
                        examples=["RELIANCE.NS"])
    date: str | None = Field(None, description="ISO date (YYYY-MM-DD); None = latest",
                             examples=["2024-03-15"])
    split: str | None = Field(None, description="Prepared split: test|validation|train")
    threshold: float | None = Field(None, ge=-0.1, le=0.1,
                                    description="Signal threshold on predicted return")
    prefer_dl: bool = Field(False, description="Use the DL model if deployed")

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        """Uppercase + basic format check. Catches 'reliance' -> 'RELIANCE.NS'.

        Accepts ticker-like strings (letters, dots, digits). This is a sanity
        check, not a full exchange-symbol validator — the registry's lookup is
        the source of truth for whether a model exists.
        """
        v = v.strip().upper()
        if not v or not all(c.isalnum() or c in ".-" for c in v):
            raise ValueError("symbol must be alphanumeric with optional . or -")
        return v


class PredictionResult(BaseModel):
    """One prediction (the frozen response contract)."""
    symbol: str
    date: str
    model: str | None = None
    model_type: str = "ml"
    predicted_return: float
    signal: str
    realised_return: float | None = None
    correct_direction: bool | None = None
    latency_ms: float | None = None


class PredictResponse(BaseModel):
    """Response for POST /predict."""
    prediction: PredictionResult
    request_id: str | None = None


# -------------------------------------------------- batch

class BatchPredictRequest(BaseModel):
    """Request body for POST /predict/batch."""
    symbols: list[str] = Field(..., min_length=1, max_length=100,
                               description="Tickers to predict",
                               examples=[["RELIANCE.NS", "TCS.NS"]])
    date: str | None = None
    split: str | None = None
    threshold: float | None = Field(None, ge=-0.1, le=0.1)
    prefer_dl: bool = False

    @field_validator("symbols")
    @classmethod
    def _normalize_symbols(cls, v: list[str]) -> list[str]:
        out = []
        for s in v:
            s = s.strip().upper()
            if not s or not all(c.isalnum() or c in ".-" for c in s):
                raise ValueError(f"invalid symbol: {s!r}")
            out.append(s)
        # Dedupe while preserving order.
        seen, dedup = set(), []
        for s in out:
            if s not in seen:
                seen.add(s); dedup.append(s)
        return dedup


class BatchPredictResponse(BaseModel):
    """Response for POST /predict/batch. Per-symbol errors are inline."""
    predictions: dict[str, Any]
    n_total: int
    n_success: int
    n_errors: int
    request_id: str | None = None


# -------------------------------------------------- models listing

class ModelInfo(BaseModel):
    """One deployed model's public metadata (GET /models)."""
    symbol: str
    model_name: str
    model_type: str
    source: str
    model_version: str
    target_col: str
    arch_type: str | None = None
    n_features: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    count: int
