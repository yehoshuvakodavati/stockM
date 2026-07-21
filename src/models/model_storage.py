"""
StockM v1.0 - Phase 5, Lesson 11
Model Storage & Versioning
===========================

Saves trained models (joblib) alongside versioned metadata (JSON) so any
saved artifact is self-describing and reproducible: you can tell, months
later, exactly which data, target, features, hyperparameters, and metrics
produced a given model file.

Layout
------
    models/saved_models/<SYMBOL>/
        <model_name>.joblib                  # the fitted estimator
        <model_name>_metadata.json           # full provenance + metrics
        best_model.json                      # name of the deployed model

Versioning
----------
Every metadata record carries:
  - model_version: a project-wide version string (from config)
  - trained_at:    ISO timestamp
  - symbol, target_col, feature_names, n_features
  - dataset: split row counts + date ranges (the exact data the model saw)
  - hyperparameters, training_time_s
  - metrics: validation + test
  - scaler reference: path to the prepared scaler params (for live inference)
  - stockm_version: codebase version tag

This is the minimum bar for MLOps: an artifact must be traceable to its
data and config, or it cannot be trusted in production.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"

logger = logging.getLogger("stockm.models.storage")

STOCKM_VERSION = "1.0"


def _safe_filename(symbol: str) -> str:
    return symbol.replace(".", "_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_metadata(
    *,
    model_name: str,
    symbol: str,
    target_col: str,
    feature_names: list[str],
    hyperparameters: dict[str, Any],
    training_time_s: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    dataset_info: dict[str, Any],
    scaler_ref: str | None = None,
    is_best: bool = False,
    model_version: str = "baseline_v1",
) -> dict[str, Any]:
    """Assemble the full provenance/metadata record for a trained model."""
    return {
        "model_name": model_name,
        "symbol": symbol,
        "model_version": model_version,
        "stockm_version": STOCKM_VERSION,
        "trained_at": _now_iso(),
        "target_col": target_col,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "hyperparameters": _jsonable(hyperparameters),
        "training_time_s": round(float(training_time_s), 4),
        "metrics": {
            "validation": _jsonable(val_metrics),
            "test": _jsonable(test_metrics),
        },
        "dataset": dataset_info,
        "scaler_ref": scaler_ref,
        "is_best": is_best,
    }


def _jsonable(obj: Any) -> Any:
    """Coerce numpy/scipy types to JSON-serialisable Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (isinstance(v, float) and v != v) else v  # NaN -> None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        return None if obj != obj else obj  # NaN -> None
    return obj


def save_model(
    model,
    metadata: dict[str, Any],
    symbol: str,
    model_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Persist a fitted model + its metadata. Returns the model file path.

    ``output_dir`` is the BASE saved-models directory; the symbol's
    subdirectory is ALWAYS appended so models for different tickers never
    collide (without this, 50 tickers' ``random_forest.joblib`` would
    overwrite each other in one flat folder).
    """
    base = output_dir or SAVED_MODELS_DIR
    out_dir = base / _safe_filename(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_name}.joblib"
    meta_path = out_dir / f"{model_name}_metadata.json"

    joblib.dump(model, model_path)
    meta_path.write_text(
        json.dumps(_jsonable(metadata), indent=2), encoding="utf-8"
    )
    logger.info("saved %s / %s", symbol, model_name)
    return model_path


def mark_best(symbol: str, model_name: str, output_dir: Path | None = None) -> Path:
    """Record which model is deployed for a symbol (read by the predictor)."""
    base = output_dir or SAVED_MODELS_DIR
    out_dir = base / _safe_filename(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_model.json"
    path.write_text(
        json.dumps(
            {"symbol": symbol, "best_model": model_name, "chosen_by": "val_rmse",
             "updated_at": _now_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_model(
    symbol: str, model_name: str | None = None, output_dir: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    """Load a fitted model + metadata. If model_name is None, loads the best.

    Args:
        symbol:     Ticker.
        model_name: Specific model, or None to load the deployed (best) one.
        output_dir: Override the default saved-models directory.

    Returns:
        (model, metadata).
    """
    base = output_dir or SAVED_MODELS_DIR
    out_dir = base / _safe_filename(symbol)
    if model_name is None:
        best_path = out_dir / "best_model.json"
        if not best_path.exists():
            raise FileNotFoundError(
                f"No best_model.json for {symbol} at {best_path}. Train first."
            )
        model_name = json.loads(best_path.read_text(encoding="utf-8"))["best_model"]

    model_path = out_dir / f"{model_name}.joblib"
    meta_path = out_dir / f"{model_name}_metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, metadata
