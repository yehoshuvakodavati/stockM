"""
StockM v1.0 - Phase 6, Lesson 9
Model Selection: Baseline vs Optimized
=======================================

For each ticker, compares the Phase 5 baseline (default hyperparameters)
against the Phase 6 optimized model (tuned). The honest question is not
"which has the lowest number" but "did tuning actually help, and is the
gain real or just validation-set noise?"

Selection rule (anti-overfit-the-validation)
--------------------------------------------
The optimized model must BEAT the baseline on validation RMSE by a small
margin to be worth deploying. If it only ties (within a tolerance), we
prefer the SIMPLER / baseline model - because an optimized model that
barely beats a default is likely overfit to the validation set and will
not generalise. This is the senior principle: don't deploy complexity
that hasn't earned its keep.

Both numbers are still validation-derived; the TEST set is reported for
honesty but never used to choose.
"""

from __future__ import annotations

from typing import Any


# A model must beat the baseline val RMSE by at least this fraction to be
# considered a "real" improvement (guards against validation-set noise).
IMPROVEMENT_TOLERANCE = 0.005  # 0.5% relative improvement


def compare_models(
    baseline_val_rmse: float,
    baseline_test_rmse: float,
    baseline_model: str,
    optimized_val_rmse: float,
    optimized_test_rmse: float,
    optimized_model: str,
    directional_accuracy: float | None = None,
) -> dict[str, Any]:
    """Build a baseline-vs-optimized comparison row + a deploy recommendation.

    Args:
        baseline_*:    Phase 5 baseline (default-params) metrics.
        optimized_*:   Phase 6 tuned model metrics.
        directional_accuracy: optimized model's test directional accuracy.

    Returns:
        Dict with the comparison numbers and a ``deploy`` decision +
        ``reason``.
    """
    rel_improvement = (
        (baseline_val_rmse - optimized_val_rmse) / baseline_val_rmse
        if baseline_val_rmse else 0.0
    )
    beats_baseline = rel_improvement > IMPROVEMENT_TOLERANCE

    if beats_baseline:
        deploy = "optimized"
        reason = (
            f"optimized val RMSE {optimized_val_rmse:.6f} beats baseline "
            f"{baseline_val_rmse:.6f} by {rel_improvement*100:.2f}% (>tol "
            f"{IMPROVEMENT_TOLERANCE*100:.1f}%). Deploy optimized."
        )
    else:
        # Barely-beats-or-ties: prefer the simpler baseline.
        deploy = "baseline"
        reason = (
            f"optimized val RMSE {optimized_val_rmse:.6f} does NOT beat "
            f"baseline {baseline_val_rmse:.6f} by >{IMPROVEMENT_TOLERANCE*100:.1f}% "
            f"(only {rel_improvement*100:.2f}%). Keep the simpler baseline."
        )

    return {
        "baseline_model": baseline_model,
        "baseline_val_rmse": round(baseline_val_rmse, 6),
        "baseline_test_rmse": round(baseline_test_rmse, 6),
        "optimized_model": optimized_model,
        "optimized_val_rmse": round(optimized_val_rmse, 6),
        "optimized_test_rmse": round(optimized_test_rmse, 6),
        "relative_val_improvement_pct": round(rel_improvement * 100, 3),
        "directional_accuracy": (
            round(directional_accuracy, 4) if directional_accuracy is not None else None
        ),
        "beats_baseline": bool(beats_baseline),
        "deploy": deploy,
        "reason": reason,
    }


def select_best_optimized(
    per_model_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Among several optimized models, pick the one with lowest CV RMSE.

    Args:
        per_model_results: {model_name: {best_cv_rmse, best_estimator, ...}}.

    Returns:
        The chosen model's result dict, or {} if none.
    """
    valid = {k: v for k, v in per_model_results.items()
             if v.get("best_cv_rmse") is not None}
    if not valid:
        return {}
    best_name = min(valid, key=lambda k: valid[k]["best_cv_rmse"])
    return {"model": best_name, **valid[best_name]}
