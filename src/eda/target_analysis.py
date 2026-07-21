"""
StockM v1.0 - Phase 4, Lesson 7
Target Analysis
===============

The target is what the model predicts. Understanding its distribution is
non-negotiable: a model trained on an imbalanced target learns "always guess
the majority class" and reports high accuracy while being worthless.

Targets produced in Phase 3
---------------------------
- target_next_return : continuous (regression). Mean ~0, fat-tailed.
- target_return_5d   : continuous, multi-horizon.
- target_direction   : binary 1/0 (UP/DOWN next day).
- target_signal      : 3-class Buy(2)/Hold(1)/Sell(0).

Class imbalance
---------------
For a binary target, imbalance ratio = max_class / min_class. Up to ~1.5:1
is mild (handle with class weights). Beyond 5:1 needs resampling
(SMOTE / undersampling) or threshold tuning. For our NIFTY-50 next-day
direction target we expect ~50/50 (mild) - but the 3-class signal is
typically 50/25/25, which is moderate and worth weighting.

Recommendations emitted
-----------------------
The module returns the imbalance ratio, per-class counts, and a recommended
handling strategy string that the training session can consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _imbalance_strategy(counts: pd.Series) -> str:
    """Pick a strategy label from the class balance."""
    ratio = counts.max() / max(counts.min(), 1)
    if ratio < 1.5:
        return "balanced - no action needed"
    if ratio < 5:
        return "mild imbalance - use class_weight='balanced' in the model"
    return "severe imbalance - resample (SMOTE/undersample) + class weights"


def target_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Analyse regression + classification targets for distribution/imbalance.

    Args:
        df: Feature frame containing the target_* columns.

    Returns:
        Dict with per-target stats (regression) and per-target class balance
        (classification) + recommended handling.
    """
    out: dict[str, Any] = {"regression": {}, "classification": {}}

    reg_targets = [c for c in df.columns
                   if c.startswith("target_") and pd.api.types.is_float_dtype(df[c])]
    cls_targets = [c for c in df.columns
                   if c.startswith("target_") and not pd.api.types.is_float_dtype(df[c])]

    for t in reg_targets:
        s = df[t].dropna()
        out["regression"][t] = {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "skew": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "pct_positive": round(float((s > 0).mean()) * 100, 2),
        }

    for t in cls_targets:
        s = df[t].dropna().astype(int)
        counts = s.value_counts().sort_index()
        out["classification"][t] = {
            "count": int(len(s)),
            "class_counts": {int(k): int(v) for k, v in counts.items()},
            "class_pct": {int(k): round(float(v) / len(s) * 100, 2)
                          for k, v in counts.items()},
            "imbalance_ratio": round(float(counts.max() / max(counts.min(), 1)), 3),
            "majority_class": int(counts.idxmax()),
            "recommendation": _imbalance_strategy(counts),
        }

    return out
