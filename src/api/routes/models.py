"""Models route — list deployed models (GET /models)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_registry
from api.model_registry import ModelRegistry
from api.schemas.predict import ModelInfo, ModelsResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsResponse, summary="List deployed models")
def list_models(
    model_type: str | None = Query(None, description="Filter: ml | dl"),
    registry: ModelRegistry = Depends(get_registry),
) -> ModelsResponse:
    """List all deployed models with their metadata.

    Optional ``model_type`` filter (ml/dl) lets a client ask for just the DL
    models, for example. Returns the symbol, model name, version, metrics, and
    feature count for each.
    """
    entries = registry.list_entries()
    if model_type:
        entries = [e for e in entries if e.model_type == model_type]
    models = [
        ModelInfo(
            symbol=e.symbol, model_name=e.model_name, model_type=e.model_type,
            source=e.source, model_version=e.model_version, target_col=e.target_col,
            arch_type=e.arch_type, n_features=len(e.feature_names), metrics=e.metrics,
        )
        for e in entries
    ]
    return ModelsResponse(models=models, count=len(models))
