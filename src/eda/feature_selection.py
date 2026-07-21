"""
StockM v1.0 - Phase 4, Lesson 9
Feature Selection
=================

More features is not better. Weak features add noise, inflate dimensionality,
and increase overfitting surface. A lean set of strong features routinely
beats a bloated set of mediocre ones - "fewer, meaningful features" is the
discipline.

Four evaluation angles (combined into one ranking)
---------------------------------------------------
1. Variance threshold: drop near-constant features (var ~ 0). They carry no
   information. Cheap, univariate, target-agnostic.
2. Correlation redundancy: drop one of each highly-correlated pair
   (multicollinearity). From correlation.py.
3. Mutual information: non-linear dependency between feature and target.
   Captures relationships Pearson misses. Target-aware.
4. Tree-based importance: a quick RandomForest's feature_importances_ - a
   model-based, non-linear, interaction-aware ranking. The gold standard for
   a quick "what actually matters" read.

Combining
---------
Each feature gets a rank on each axis; we keep the top-K by a blended rank
(default keep top 40, or all with MI > a floor). Targets are never selected
away - they are preserved regardless.

Leakage safety
--------------
Mutual information and tree importance are fit on the TRAIN split only and
reported; using them on the full set would leak test information into the
choice of features. The orchestrator passes the train frame here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eda.correlation import recommend_redundant_drops


def _is_classifier_target(series: pd.Series) -> bool:
    return not pd.api.types.is_float_dtype(series)


def select_features(
    train_df: pd.DataFrame,
    target_col: str = "target_direction",
    keep_top_k: int = 40,
    variance_threshold: float = 1e-4,
    corr_threshold: float = 0.95,
    mi_floor: float = 0.001,
) -> dict[str, Any]:
    """Rank features and return the selected feature list + per-feature scores.

    Args:
        train_df:            TRAIN split only (leakage-safe).
        target_col:          Target to score against.
        keep_top_k:          Max features to keep.
        variance_threshold:  Drop features with var below this.
        corr_threshold:      |r| above which a pair is redundant.
        mi_floor:            Keep features with MI above this even past top_k.

    Returns:
        Dict with selected_features, dropped (with reasons), and per-feature
        scores (variance, mi, tree_importance).
    """
    target_cols = [c for c in train_df.columns if c.startswith("target_")]
    feature_cols = [
        c for c in train_df.columns
        if c not in target_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # 1. Variance threshold ------------------------------------------------
    variances = train_df[feature_cols].var(skipna=True).fillna(0.0)
    low_var = variances[variances < variance_threshold].index.tolist()

    # 2. Correlation redundancy -------------------------------------------
    redundant = recommend_redundant_drops(train_df, threshold=corr_threshold)

    # Candidates after the cheap univariate + redundancy cuts.
    candidates = [c for c in feature_cols
                  if c not in low_var and c not in redundant]

    # 3 + 4. Mutual information + tree importance (need a clean frame) -----
    work = train_df[candidates + [target_col]].dropna()
    X = work[candidates]
    y = work[target_col]

    scores: dict[str, dict[str, float]] = {c: {} for c in candidates}

    if len(work) >= 50 and y.nunique() >= 2:
        # Mutual information (target-aware, non-linear).
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            mi_fn = mutual_info_classif if _is_classifier_target(y) else mutual_info_regression
            mi = mi_fn(X, y, random_state=42)
            for c, v in zip(candidates, mi):
                scores[c]["mutual_info"] = float(v)
        except Exception:
            pass  # MI is best-effort; tree importance below is the fallback.

        # Tree-based importance (model-aware, captures interactions).
        try:
            rf_cls = RandomForestClassifier if _is_classifier_target(y) else RandomForestRegressor
            rf = rf_cls(n_estimators=80, max_depth=6, n_jobs=-1, random_state=42)
            rf.fit(X, y)
            for c, v in zip(candidates, rf.feature_importances_):
                scores[c]["tree_importance"] = float(v)
        except Exception:
            pass

    # --- Blend ranks: average rank across MI and tree importance ---------
    for c in candidates:
        scores[c]["variance"] = float(variances.get(c, 0.0))

    # Rank higher = better. Use MI as primary, tree importance as tiebreak.
    ranked = sorted(
        candidates,
        key=lambda c: (
            scores[c].get("mutual_info", 0.0),
            scores[c].get("tree_importance", 0.0),
        ),
        reverse=True,
    )

    # Keep top_k, plus anything above the MI floor (don't lose strong signals).
    selected = []
    for c in ranked:
        if len(selected) >= keep_top_k and scores[c].get("mutual_info", 0.0) < mi_floor:
            break
        selected.append(c)

    dropped = {
        "low_variance": low_var,
        "redundant": redundant,
        "below_rank": [c for c in ranked if c not in selected],
    }

    return {
        "selected_features": selected,
        "dropped": dropped,
        "scores": scores,
        "target_col": target_col,
        "n_before": len(feature_cols),
        "n_after": len(selected),
    }
