"""
StockM v1.0 - Phase 9, Lesson 5
Prediction Service
===================

The business layer: turns an HTTP request's intent into a prediction by
calling the existing, verified prediction engine (models.prediction /
deep_learning.inference). Knows NOTHING about HTTP — it takes plain args and
returns plain dicts. This is what makes it reusable from the API, scripts, and
tests without coupling to FastAPI.

Why a service layer
-------------------
A route that calls ``models.prediction.predict_single_stock`` directly couples
HTTP to the prediction engine's evolving signature. The service layer freezes a
stable internal contract (predict_single_stock(symbol) -> dict) and absorbs
engine changes (new args, different return keys) in one place. It also adds
cross-cutting concerns that belong to "serving", not "predicting": timing,
logging, and (Lesson 10) metrics emission.

Three operations (mirror the roadmap):
    predict(symbol, ...)           -> single prediction (the common case)
    predict_single_stock(symbol)   -> alias; explicit single-stock predict
    predict_batch(symbols)         -> many symbols, resilient (one failure
                                      doesn't sink the batch)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from api.model_registry import ModelRegistry

logger = logging.getLogger("stockm.api.service")


class PredictionService:
    """Wraps the prediction engine with serving concerns (timing, logging).

    Built once at startup (in main.py's lifespan) with the startup-loaded
    registry, and stored on app.state. The DI layer hands it to routes.
    """

    def __init__(self, registry: ModelRegistry, default_split: str = "test",
                 default_threshold: float = 0.0) -> None:
        self.registry = registry
        self.default_split = default_split
        self.default_threshold = default_threshold

    # -------------------------------------------------- single prediction

    def predict_single_stock(
        self,
        symbol: str,
        date: str | None = None,
        split: str | None = None,
        threshold: float | None = None,
        prefer_dl: bool = False,
    ) -> dict[str, Any]:
        """Predict next-day return + signal for one symbol.

        Delegates to the Phase 7 predictor (ML by default, DL if prefer_dl and
        a DL model is deployed). Times the call and logs it. Returns the
        predictor's dict augmented with ``latency_ms``.

        Args:
            symbol:    Ticker, e.g. "RELIANCE.NS".
            date:      ISO date to predict for; None = latest in split.
            split:     Prepared split to predict on (default from settings).
            threshold: Signal threshold (default from settings).
            prefer_dl: Use the DL model if available.

        Returns:
            Dict with symbol, date, predicted_return, signal, model provenance.

        Raises:
            KeyError: if no model is deployed for the symbol.
        """
        # Verify deployment before calling the engine (fast fail, clear error).
        entry = self.registry.get_entry(symbol)
        if entry is None and not prefer_dl:
            raise KeyError(f"no deployed model for {symbol!r}")

        split_ = split or self.default_split
        thr = threshold if threshold is not None else self.default_threshold

        t0 = time.perf_counter()
        if prefer_dl:
            from deep_learning.inference import predict_single_stock as _predict
        else:
            from models.prediction import predict_single_stock as _predict
        result = _predict(symbol, date=date, split=split_, threshold=thr)
        latency_ms = (time.perf_counter() - t0) * 1000

        result["latency_ms"] = round(latency_ms, 2)
        result["model_type"] = "dl" if prefer_dl else "ml"
        logger.info(
            "predict %s date=%s signal=%s ret=%.6f latency=%.1fms",
            symbol, result.get("date"), result.get("signal"),
            result.get("predicted_return", 0.0), latency_ms,
        )
        return result

    # Alias matching the roadmap's naming.
    def predict(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        """Alias for :meth:`predict_single_stock` (the common single call)."""
        return self.predict_single_stock(symbol, **kwargs)

    # -------------------------------------------------- batch

    def predict_batch(
        self,
        symbols: list[str],
        date: str | None = None,
        split: str | None = None,
        threshold: float | None = None,
        prefer_dl: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Predict for many symbols. One failure never sinks the batch.

        Returns {symbol: prediction_or_error}. A symbol with no deployed model
        or a prediction error gets an ``error`` entry, not an exception — so a
        50-symbol batch with one bad ticker returns 49 results + 1 error.
        """
        t0 = time.perf_counter()
        out: dict[str, Any] = {}
        for sym in symbols:
            try:
                out[sym] = self.predict_single_stock(
                    sym, date=date, split=split, threshold=threshold,
                    prefer_dl=prefer_dl,
                )
            except KeyError as e:
                out[sym] = {"symbol": sym, "error": str(e)}
                logger.warning("batch predict: %s", e)
            except Exception as e:  # noqa: BLE001 - batch must continue
                out[sym] = {"symbol": sym, "error": str(e)}
                logger.warning("batch predict failed for %s: %s", sym, e)
        logger.info(
            "batch predict: %d symbols, %d ok, %d errors, %.1fms",
            len(symbols),
            sum(1 for v in out.values() if "error" not in v),
            sum(1 for v in out.values() if "error" in v),
            (time.perf_counter() - t0) * 1000,
        )
        return out
