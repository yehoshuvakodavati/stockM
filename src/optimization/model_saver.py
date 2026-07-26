"""
StockM v1.0 - Phase 6, Lesson 12
Save Optimized Models
=====================

Persists the optimized (tuned) model alongside full provenance so the
production predictor can load it and reproduce the exact config that won.

Layout
------
    models/optimized/<SYMBOL>/
        <model_name>.joblib          # the tuned estimator
        <model_name>_metadata.json   # full provenance + metrics
        best_optimized.json          # which model is deployed for this ticker

Metadata (versioned, reproducible)
----------------------------------
  model_name, model_version, optimization_method, best_hyperparameters,
  scaler_ref, selected_features, dataset_version (symbol + rows + dates),
  cv (method + n_splits), metrics (CV + test + directional accuracy),
  training_time_s, tuned_at (ISO), seed, stockm_version.

This mirrors model_storage.py (Phase 5) but lives under models/optimized/
so tuned and untuned artifacts never collide, and the predictor can prefer
the tuned one.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger("stockm.optimization.saver")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZED_DIR = PROJECT_ROOT / "models" / "optimized"
STOCKM_VERSION = "1.0"


def _safe_filename(symbol: str) -> str:
    return symbol.replace(".", "_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, float):
        return None if obj != obj else obj
    return obj


def build_optimized_metadata(
    *,
    model_name: str,
    symbol: str,
    optimization_method: str,
    best_hyperparameters: dict[str, Any],
    feature_names: list[str],
    scaler_ref: str | None,
    dataset_info: dict[str, Any],
    cv_info: dict[str, Any],
    metrics: dict[str, Any],
    training_time_s: float,
    seed: int = 42,
    model_version: str = "optimized_v1",
) -> dict[str, Any]:
    """Assemble the provenance record for a tuned model."""
    return {
        "model_name": model_name,
        "symbol": symbol,
        "model_version": model_version,
        "optimization_method": optimization_method,
        "best_hyperparameters": _jsonable(best_hyperparameters),
        "selected_features": feature_names,
        "n_features": len(feature_names),
        "scaler_ref": scaler_ref,
        "dataset_version": dataset_info,
        "cv": cv_info,
        "metrics": _jsonable(metrics),
        "training_time_s": round(float(training_time_s), 4),
        "seed": int(seed),
        "tuned_at": _now_iso(),
        "stockm_version": STOCKM_VERSION,
    }


def save_optimized_model(
    model, metadata: dict[str, Any], symbol: str, model_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Persist a tuned model + its metadata. Returns the model file path."""
    base = output_dir or OPTIMIZED_DIR
    out_dir = base / _safe_filename(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_name}.joblib"
    meta_path = out_dir / f"{model_name}_metadata.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    logger.info("saved optimized %s / %s", symbol, model_name)
    return model_path


def mark_best_optimized(
    symbol: str, model_name: str, output_dir: Path | None = None,
) -> Path:
    """Record which tuned model is deployed for a symbol."""
    base = output_dir or OPTIMIZED_DIR
    out_dir = base / _safe_filename(symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_optimized.json"
    path.write_text(
        json.dumps(
            {"symbol": symbol, "best_model": model_name, "chosen_by": "cv_rmse",
             "updated_at": _now_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_optimized_model(
    symbol: str, model_name: str | None = None, output_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a tuned model + metadata. If model_name is None, load the deployed one."""
    base = output_dir or OPTIMIZED_DIR
    out_dir = base / _safe_filename(symbol)
    if model_name is None:
        best_path = out_dir / "best_optimized.json"
        if not best_path.exists():
            raise FileNotFoundError(f"No best_optimized.json for {symbol} at {best_path}.")
        model_name = json.loads(best_path.read_text(encoding="utf-8"))["best_model"]

    model_path = out_dir / f"{model_name}.joblib"
    meta_path = out_dir / f"{model_name}_metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Optimized model not found: {model_path}")
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return model, metadata
