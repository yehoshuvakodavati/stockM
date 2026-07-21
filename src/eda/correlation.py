"""
StockM v1.0 - Phase 4, Lesson 5
Correlation Analysis
====================

Correlation measures how tightly two features move together. In feature
engineering this has two uses:

1. Feature-vs-target correlation: a weak signal that a feature is predictive
   (in finance, |r| of 0.02-0.08 is typical; >0.2 is suspicious - likely
   leakage).
2. Feature-vs-feature correlation (multicollinearity): redundant features.
   Two columns with |r| > 0.95 carry almost the same information; feeding
   both to a linear model destabilises coefficients, and to any model wastes
   capacity.

Pearson vs Spearman
-------------------
- Pearson: linear relationship. Sensitive to outliers (one crash day can
  distort it). Good for normally-distributed, linear pairs.
- Spearman: rank-based monotonic relationship. Robust to outliers and
  non-linear-but-monotonic pairs. Often the safer choice for finance.

Redundancy recommendation
-------------------------
For each cluster of highly-correlated features we recommend keeping ONE
representative (the one with highest variance, or highest target correlation
once a target is supplied) and dropping the rest. The actual dropping is
done by feature_selection.py; this module only reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def correlation_analysis(
    df: pd.DataFrame,
    threshold: float = 0.95,
    target_col: str | None = None,
    top_n_pairs: int = 30,
    top_n_target: int = 20,
) -> dict[str, Any]:
    """Compute Pearson + Spearman matrices and flag redundancy / target signal.

    Args:
        df:          Feature frame.
        threshold:   |r| above which a feature pair is flagged as redundant.
        target_col:  Optional target column to rank features by predictiveness.
        top_n_pairs: Max redundant pairs to list.
        top_n_target:Max feature-vs-target correlations to list.

    Returns:
        Dict with redundant_pairs, target_correlations, and the matrices'
        shapes (the full matrices are large; we ship summaries, not N*N dicts).
    """
    numeric = df.select_dtypes(include="number")

    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")

    # --- Redundant pairs (Pearson upper triangle) --------------------------
    upper = pearson.where(np.triu(np.ones(pearson.shape), k=1).astype(bool))
    stacked = upper.abs().stack()
    high = stacked[stacked >= threshold].sort_values(ascending=False)

    redundant: list[dict[str, Any]] = []
    for (a, b), val in high.head(top_n_pairs).items():
        redundant.append({
            "feature_a": a, "feature_b": b,
            "pearson": round(float(pearson.loc[a, b]), 4),
            "spearman": round(float(spearman.loc[a, b]), 4),
        })

    # --- Feature-vs-target correlations -----------------------------------
    target_corr: list[dict[str, Any]] = []
    if target_col and target_col in numeric.columns:
        t = numeric[target_col]
        for col in numeric.columns:
            if col == target_col:
                continue
            pair = pd.concat([numeric[col], t], axis=1).dropna()
            if len(pair) < 5 or pair[col].nunique() < 2:
                continue
            target_corr.append({
                "feature": col,
                "pearson": round(float(pair[col].corr(pair[target_col])), 4),
                "spearman": round(
                    float(pair[col].corr(pair[target_col], method="spearman")), 4
                ),
            })
        # Rank by absolute Pearson - the usual "most predictive" ordering.
        target_corr.sort(key=lambda d: abs(d["pearson"]), reverse=True)
        target_corr = target_corr[:top_n_target]

    return {
        "feature_count": int(numeric.shape[1]),
        "redundant_pair_count": int(len(high)),
        "redundant_pairs": redundant,
        "target_correlations": target_corr,
        "threshold": threshold,
    }


def recommend_redundant_drops(
    df: pd.DataFrame,
    threshold: float = 0.95,
    keep_targets: bool = True,
) -> list[str]:
    """Return a list of columns recommended for removal as redundant.

    Greedy strategy: for each highly-correlated pair, drop the column that
    appears in the most redundant pairs (i.e. the more "hub-like" feature's
    partner). This is a heuristic; feature_selection.py combines it with
    variance and mutual information for the final cut.
    """
    numeric = df.select_dtypes(include="number")
    if keep_targets:
        numeric = numeric[[c for c in numeric.columns if not c.startswith("target_")]]

    corr = numeric.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Count how many high-correlation partners each column has.
    to_drop: set[str] = set()
    for col in upper.columns:
        # If this column is highly correlated with any *not-yet-dropped*
        # earlier column, mark it for dropping (keep the first seen).
        partners = upper.index[upper[col] >= threshold].tolist()
        for p in partners:
            if p not in to_drop and col not in to_drop:
                # Keep the one with greater variance (more information).
                if numeric[col].var() >= numeric[p].var():
                    to_drop.add(p)
                else:
                    to_drop.add(col)
    return sorted(to_drop)
