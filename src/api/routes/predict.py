"""Predict routes — POST /predict and POST /predict/batch."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.config import get_settings
from api.dependencies import get_prediction_service
from api.errors import ModelNotAvailableError, PredictionError
from api.schemas.predict import (
    BatchPredictRequest, BatchPredictResponse,
    PredictRequest, PredictResponse, PredictionResult,
)
from api.security import rate_limit, require_api_key
from api.services.prediction_service import PredictionService

router = APIRouter(tags=["prediction"])

# In production, protect prediction endpoints with auth + rate limiting.
# In dev, leave them open for convenience (auth disabled when no API_KEY set).
_prod_deps = []
if get_settings().is_production:
    _prod_deps = [Depends(require_api_key), Depends(rate_limit)]


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict for one symbol",
    dependencies=_prod_deps,
    description=(
        "Generate a BUY/HOLD/SELL prediction for a single symbol. Returns the "
        "predicted next-day return, the signal, and the model that produced it. "
        "A symbol with no deployed model returns 404.\n\n"
        "**Example request:**\n```json\n{\"symbol\": \"RELIANCE.NS\"}\n```\n\n"
        "**Example response:**\n```json\n"
        "{\"prediction\": {\"symbol\": \"RELIANCE.NS\", \"date\": \"2024-03-15\", "
        "\"predicted_return\": 0.0023, \"signal\": \"BUY\", \"model_type\": \"ml\", "
        "\"latency_ms\": 12.4}}\n```"
    ),
    responses={
        404: {"description": "No deployed model for the symbol",
              "content": {"application/json": {"example": {
                  "detail": "no deployed model for 'UNKNOWN.NS'",
                  "error_code": "MODEL_NOT_FOUND"}}}},
        422: {"description": "Validation error (bad symbol/threshold)"},
        503: {"description": "Prediction engine failure"},
    },
)
def predict(
    body: PredictRequest,
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    """Generate a BUY/HOLD/SELL prediction for a single symbol.

    Returns the predicted next-day return, the signal, and the model that
    produced it. A symbol with no deployed model returns 404.
    """
    try:
        result = service.predict_single_stock(
            body.symbol, date=body.date, split=body.split,
            threshold=body.threshold, prefer_dl=body.prefer_dl,
        )
    except KeyError as e:
        raise ModelNotAvailableError(str(e)) from e
    except Exception as e:  # noqa: BLE001 - engine failure -> 503
        raise PredictionError(f"prediction failed for {body.symbol}: {e}") from e

    return PredictResponse(
        prediction=PredictionResult(**result),
        request_id=getattr(request.state, "request_id", None),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Predict for multiple symbols",
    dependencies=_prod_deps,
    description=(
        "Generate predictions for many symbols at once. Per-symbol errors are "
        "returned inline (not raised) so a 50-symbol batch with one bad ticker "
        "returns 49 results + 1 error entry.\n\n"
        "**Example request:**\n```json\n{\"symbols\": [\"RELIANCE.NS\", \"TCS.NS\"]}\n```"
    ),
)
def predict_batch(
    body: BatchPredictRequest,
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> BatchPredictResponse:
    """Generate predictions for many symbols at once.

    Per-symbol errors are returned inline (not raised) so a 50-symbol batch
    with one bad ticker returns 49 results + 1 error entry.
    """
    results = service.predict_batch(
        body.symbols, date=body.date, split=body.split,
        threshold=body.threshold, prefer_dl=body.prefer_dl,
    )
    n_err = sum(1 for v in results.values() if "error" in v)
    return BatchPredictResponse(
        predictions=results,
        n_total=len(results),
        n_success=len(results) - n_err,
        n_errors=n_err,
        request_id=getattr(request.state, "request_id", None),
    )
