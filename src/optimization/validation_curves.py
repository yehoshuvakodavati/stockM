"""
StockM v1.0 - Phase 6, Lesson 8
Overfitting, Underfitting, and Diagnostic Curves
=================================================

Two curves diagnose the bias-variance state of a model:

Learning curve (error vs training-set size)
    Train + validation error plotted against the number of training rows.
    - High-bias (underfit): both curves plateau early and HIGH, close together.
      Fix: more capacity (deeper trees, more features, less regularization).
    - High-variance (overfit): train error LOW, val error HIGH, gap is wide.
      Fix: more data, more regularization, simpler model.
    - Both low and converging: well-fit. The plateau is the irreducible-noise
      floor - in finance this floor is HIGH, so don't expect zero.

Validation curve (error vs a single hyperparameter)
    Train + validation error vs a hyperparameter value (e.g. max_depth).
    - As capacity rises (depth up): train error falls, val error falls then
      rises - the U-turn is the overfitting onset. The val minimum is the
      sweet spot (the bias-variance tradeoff point).
    - If val keeps falling at the edge, extend the range.

Both are computed with TIME-SERIES CV (not K-Fold) to stay leak-free.
Plots are saved as PNGs under reports/optimization/curves/ and the raw
curve data is saved as JSON for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import learning_curve, validation_curve

logger = logging.getLogger("stockm.optimization.curves")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURVES_DIR = PROJECT_ROOT / "reports" / "optimization" / "curves"

DEFAULT_SCORING = "neg_root_mean_squared_error"


def compute_learning_curve(
    estimator, X, y, cv,
    train_sizes=np.linspace(0.1, 1.0, 6),
    scoring: str = DEFAULT_SCORING,
) -> dict[str, Any]:
    """Compute a learning curve (error vs training-set size).

    Returns a dict with train_sizes and train/val RMSE (sign-corrected) so
    the caller can plot or save it.
    """
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=train_sizes, n_jobs=-1, error_score="raise",
    )
    # sklearn returns neg-RMSE; flip to positive RMSE for readability.
    train_rmse = -train_scores
    val_rmse = -val_scores
    return {
        "kind": "learning",
        "train_sizes": [int(s) for s in sizes],
        "train_rmse_mean": [float(v) for v in train_rmse.mean(axis=1)],
        "train_rmse_std": [float(v) for v in train_rmse.std(axis=1)],
        "val_rmse_mean": [float(v) for v in val_rmse.mean(axis=1)],
        "val_rmse_std": [float(v) for v in val_rmse.std(axis=1)],
    }


def compute_validation_curve(
    estimator, X, y, cv, param_name: str, param_range,
    scoring: str = DEFAULT_SCORING,
) -> dict[str, Any]:
    """Compute a validation curve (error vs a single hyperparameter value).

    Returns the per-value train/val RMSE so the caller can find the U-turn
    (overfitting onset).
    """
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1, error_score="raise",
    )
    train_rmse = -train_scores
    val_rmse = -val_scores
    return {
        "kind": "validation",
        "param_name": param_name,
        "param_range": [float(p) if isinstance(p, (np.floating, float)) else
                        (int(p) if isinstance(p, (np.integer, int)) else str(p))
                        for p in param_range],
        "train_rmse_mean": [float(v) for v in train_rmse.mean(axis=1)],
        "val_rmse_mean": [float(v) for v in val_rmse.mean(axis=1)],
        "val_min_at": _index_of_min(val_rmse.mean(axis=1)),
    }


def _index_of_min(arr) -> Any:
    """Return the param_range index where validation error is lowest."""
    arr = np.asarray(arr, dtype="float64")
    if arr.size == 0:
        return None
    return int(np.argmin(arr))


def save_curves(
    symbol: str, model_name: str,
    learning: dict | None, validation: dict | None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist curve data as JSON and (if matplotlib is available) PNGs.

    Returns a dict of saved paths. Plotting is best-effort: in headless
    environments the JSON is still written, so no information is lost.
    """
    out_dir = output_dir or CURVES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol.replace('.', '_')}_{model_name}"
    paths: dict[str, Path] = {}

    if learning is not None:
        jp = out_dir / f"{stem}_learning_curve.json"
        jp.write_text(json.dumps(learning, indent=2), encoding="utf-8")
        paths["learning_json"] = jp
    if validation is not None:
        jp = out_dir / f"{stem}_validation_curve.json"
        jp.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        paths["validation_json"] = jp

    _try_plot(out_dir, stem, learning=learning, validation=validation, paths=paths)
    return paths


def _try_plot(out_dir, stem, learning, validation, paths) -> None:
    """Best-effort matplotlib PNGs; never crash if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        logger.info("matplotlib unavailable; skipping curve PNGs (JSON saved).")
        return

    if learning is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(learning["train_sizes"], learning["train_rmse_mean"],
                "o-", label="train RMSE")
        ax.plot(learning["train_sizes"], learning["val_rmse_mean"],
                "s-", label="val RMSE")
        ax.set_xlabel("training rows"); ax.set_ylabel("RMSE")
        ax.set_title(f"{stem} learning curve"); ax.legend(); fig.tight_layout()
        p = out_dir / f"{stem}_learning_curve.png"; fig.savefig(p, dpi=100)
        plt.close(fig); paths["learning_png"] = p
    if validation is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(validation["param_range"], validation["train_rmse_mean"],
                "o-", label="train RMSE")
        ax.plot(validation["param_range"], validation["val_rmse_mean"],
                "s-", label="val RMSE")
        ax.set_xlabel(validation["param_name"]); ax.set_ylabel("RMSE")
        ax.set_title(f"{stem} validation curve ({validation['param_name']})")
        ax.legend(); fig.tight_layout()
        p = out_dir / f"{stem}_validation_curve.png"; fig.savefig(p, dpi=100)
        plt.close(fig); paths["validation_png"] = p
