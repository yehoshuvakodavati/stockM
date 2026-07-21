"""
StockM v1.0 - Phase 5, Lesson 10
Model Comparison
================

Puts every model's numbers side by side and recommends a winner. The
comparison is the bridge from "I trained 4 models" to "this is the one we
deploy."

Selection rule
--------------
The best model is chosen by the **lowest validation RMSE** (the split the
model never learned from, used for selection). Test metrics are reported
alongside but NEVER used to pick the winner - using test for selection leaks
future information into the deployment decision (Session 9 discipline).

Reported columns
----------------
model, training_time_s, val_rmse, val_r2, test_rmse, test_r2,
directional_accuracy (test), beats_naive (test RMSE < naive RMSE?),
advantages, weaknesses.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("stockm.models.comparison")

# Human-readable per-model qualitative notes, surfaced in the report so a
# reader gets the trade-off alongside the numbers.
MODEL_NOTES: dict[str, dict[str, str]] = {
    "linear_regression": {
        "advantages": "fastest to train/serve; fully interpretable coefficients; strong floor baseline",
        "weaknesses": "high bias: cannot model non-linearity or feature interactions",
    },
    "decision_tree": {
        "advantages": "captures non-linearity; interpretable splits",
        "weaknesses": "high variance; overfits without depth limits; unstable",
    },
    "random_forest": {
        "advantages": "low variance via bagging; robust; handles non-linearity; little tuning",
        "weaknesses": "larger memory; slower inference than linear; less interpretable",
    },
    "gradient_boosting": {
        "advantages": "often best accuracy on tabular data; captures interactions; handles NaN",
        "weaknesses": "sequential training (slower); more hyperparameters to tune",
    },
    "xgboost": {
        "advantages": "state-of-the-art boosting; regularised; fast with GPU",
        "weaknesses": "extra dependency; overfits without care",
    },
    "lightgbm": {
        "advantages": "very fast boosting; good on large data",
        "weaknesses": "extra dependency; can overfit on small data",
    },
}


def _note(name: str, key: str) -> str:
    return MODEL_NOTES.get(name, {}).get(key, "")


def build_comparison(
    results: dict[str, dict[str, Any]],
    naive_test_rmse: float,
) -> list[dict[str, Any]]:
    """Assemble a comparison row per model, sorted by validation RMSE.

    Args:
        results:        {model_name: {training_time_s, val, test, directional_accuracy}}
        naive_test_rmse: RMSE of the zero-predictor on test (the floor to beat).

    Returns:
        List of comparison rows (best first), each with advantages/weaknesses.
    """
    rows: list[dict[str, Any]] = []
    for name, r in results.items():
        val = r.get("val", {})
        test = r.get("test", {})
        test_rmse = test.get("rmse", float("inf"))
        rows.append({
            "model": name,
            "training_time_s": round(r.get("training_time_s", 0.0), 4),
            "val_rmse": round(val.get("rmse", float("nan")), 6),
            "val_r2": round(val.get("r2", float("nan")), 6),
            "test_rmse": round(test_rmse, 6),
            "test_r2": round(test.get("r2", float("nan")), 6),
            "test_mae": round(test.get("mae", float("nan")), 6),
            "directional_accuracy": round(
                r.get("directional_accuracy", float("nan")), 4
            ),
            "beats_naive": bool(test_rmse < naive_test_rmse),
            "advantages": _note(name, "advantages"),
            "weaknesses": _note(name, "weaknesses"),
        })
    # Rank by validation RMSE ascending (lower is better).
    rows.sort(key=lambda d: d["val_rmse"] if d["val_rmse"] == d["val_rmse"] else 1e9)
    return rows


def recommend_best(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the recommended model (first row - lowest val RMSE) with reasoning."""
    if not comparison:
        return {"best_model": None, "reason": "no models trained"}
    best = comparison[0]
    reason = (
        f"lowest validation RMSE ({best['val_rmse']:.6f}); "
        f"test RMSE {best['test_rmse']:.6f} "
        f"({'beats' if best['beats_naive'] else 'does NOT beat'} naive). "
        f"directional accuracy {best['directional_accuracy']:.4f}."
    )
    return {"best_model": best["model"], "reason": reason, "details": best}
