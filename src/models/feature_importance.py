"""
StockM v1.0 - Phase 5, Lesson 9
Feature Importance
==================

Which engineered features actually drive the model's predictions? This module
extracts per-model importance and builds a *consensus* ranking across models,
answering three questions:

1. Which features contribute most? (per-model and consensus)
2. Which features can be removed? (consistently unimportant across models)
3. What is the financial meaning of the top features?

Importance extraction differs by model family
----------------------------------------------
- Linear models: |coefficient| (on scaled features, so coefficients are
  comparable). Sign = direction of effect.
- Tree ensembles (RF, GBM, XGBoost, LightGBM): feature_importances_
  (impurity-based or gain-based).
- Single decision tree: feature_importances_.

Caveat: impurity importance is biased toward high-cardinality features and
ignores feature interactions; treat rankings as approximate, gospel only in
aggregate. The consensus across diverse models is more trustworthy than any
single model's ranking.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _extract_one(model, feature_names: list[str]) -> dict[str, float]:
    """Pull a raw importance vector from one fitted model.

    Returns {feature: importance}. Values are non-negative for trees, but
    may be signed for linear (we keep the sign there for interpretation and
    let the caller take abs for ranking).
    """
    # Linear models expose coef_.
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype="float64").ravel()
        return {f: float(c) for f, c in zip(feature_names, coef)}

    # Trees / ensembles expose feature_importances_.
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype="float64").ravel()
        return {f: float(v) for f, v in zip(feature_names, imp)}

    return {f: 0.0 for f in feature_names}


def get_feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Return {feature: importance} for a fitted model (signed for linear)."""
    return _extract_one(model, feature_names)


def rank_features(
    model, feature_names: list[str], top_n: int = 20
) -> list[dict[str, Any]]:
    """Rank features by absolute importance, descending.

    Args:
        model:         Fitted estimator.
        feature_names: Column names corresponding to the model's inputs.
        top_n:         Max features to return.

    Returns:
        List of {feature, importance, rank}, top first.
    """
    imp = _extract_one(model, feature_names)
    ranked = sorted(imp.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [
        {"rank": i + 1, "feature": f, "importance": round(v, 6)}
        for i, (f, v) in enumerate(ranked[:top_n])
    ]


def consensus_importance(
    models: dict[str, Any], feature_names: list[str], top_n: int = 20
) -> list[dict[str, Any]]:
    """Average rank across models -> a robust, model-agnostic ranking.

    Each model ranks features 1..N by |importance|; we average those ranks
    (lower = more important) and sort. A feature in the top-5 of every model
    ranks high; a feature that is #30 in one and #3 in another ranks middling.
    This is more reliable than averaging raw importances (which have
    incomparable scales across model families).
    """
    n = len(feature_names)
    rank_sums = {f: 0.0 for f in feature_names}
    count = 0
    for name, model in models.items():
        imp = _extract_one(model, feature_names)
        if not any(abs(v) > 0 for v in imp.values()):
            continue
        # Rank 1..N by |importance| (1 = most important).
        order = sorted(feature_names, key=lambda f: abs(imp[f]), reverse=True)
        for r, f in enumerate(order, start=1):
            rank_sums[f] += r
        count += 1

    if count == 0:
        return []
    avg_rank = {f: rank_sums[f] / count for f in feature_names}
    ranked = sorted(avg_rank.items(), key=lambda kv: kv[1])  # ascending rank
    return [
        {"rank": i + 1, "feature": f, "avg_rank": round(r, 2)}
        for i, (f, r) in enumerate(ranked[:top_n])
    ]


def recommend_drops(
    models: dict[str, Any], feature_names: list[str], bottom_frac: float = 0.2
) -> list[str]:
    """Suggest low-value features for removal (consistently unimportant).

    Returns the bottom `bottom_frac` of features by consensus average rank.
    Removing these usually costs nothing in performance and simplifies the
    model - but verify on validation before actually dropping.
    """
    cons = consensus_importance(models, feature_names, top_n=len(feature_names))
    if not cons:
        return []
    n_drop = max(1, int(len(feature_names) * bottom_frac))
    # cons is sorted most-important first; take the tail.
    return [c["feature"] for c in cons[-n_drop:]]
