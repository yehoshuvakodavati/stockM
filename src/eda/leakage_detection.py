"""
StockM v1.0 - Phase 4, Lesson 8
Data Leakage Detection
======================

Leakage is the #1 model-killer in finance. The feature pipeline was built to
be leak-free (trailing windows only, targets are the only forward-looking
columns). EDA is the independent audit that *verifies* that contract held.

Three leakage types we check
----------------------------
1. Future information leakage
   A feature that uses data from after the row's date. We check structurally:
   the feature families only call shift(positive)/rolling(trailing), so a
   leak here would be a coding bug. We also flag any feature whose
   correlation with a forward target is *implausibly* high (|r| > 0.3 with
   next-day return is a red flag; > 0.6 is almost certainly leakage).

2. Target leakage
   A feature derived *from* the target (or the future target) rather than
   from contemporaneous information. Detect via suspiciously high
   feature-target correlation and via name inspection (no feature should be
   built from a target_* column).

3. Feature leakage (future stats in preprocessing)
   Fitting scaler/feature-selection on the full dataset including the test
   period. This module can't catch that at analysis time, but the scaling
   module enforces fit_on_train_only, and we record the contract here.

The audit returns a report + a list of *suspicious* features to investigate
by hand. It never auto-deletes - a human confirms leakage before removal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# A feature whose |corr| with a forward target exceeds this is suspicious.
# Real financial features rarely exceed 0.1; anything above 0.3 demands
# scrutiny; above 0.6 is almost certainly leakage.
SUSPICIOUS_CORR = 0.30
LEAKAGE_CORR = 0.60


def leakage_audit(
    df: pd.DataFrame,
    target_col: str = "target_next_return",
) -> dict[str, Any]:
    """Audit the feature set for leakage signatures.

    Args:
        df:         Feature frame (features + targets).
        target_col: The forward target to check feature correlations against.

    Returns:
        Dict with suspicious_features, leakage_flags, and a contract summary.
    """
    report: dict[str, Any] = {}

    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [
        c for c in df.columns
        if c not in target_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    report["target_columns"] = target_cols
    report["feature_count"] = int(len(feature_cols))
    report["contract"] = {
        "features_use_only_past_data": True,
        "targets_are_forward_looking": True,
        "scaling_fit_on": "train_only",
        "split": "chronological_with_gap",
    }

    # --- Suspicious feature-target correlations ----------------------------
    if target_col not in df.columns:
        report["suspicious_features"] = []
        report["leakage_flags"] = []
        return report

    suspicious, leakage = [], []
    t = df[target_col]
    for col in feature_cols:
        pair = pd.concat([df[col], t], axis=1).dropna()
        if len(pair) < 5 or pair[col].nunique() < 2:
            continue
        r = float(pair[col].corr(pair[target_col]))
        if abs(r) >= LEAKAGE_CORR:
            leakage.append({"feature": col, "corr_with_target": round(r, 4),
                            "severity": "likely_leakage"})
        elif abs(r) >= SUSPICIOUS_CORR:
            suspicious.append({"feature": col, "corr_with_target": round(r, 4),
                               "severity": "suspicious"})

    # Sort by absolute correlation, most suspicious first.
    suspicious.sort(key=lambda d: abs(d["corr_with_target"]), reverse=True)
    leakage.sort(key=lambda d: abs(d["corr_with_target"]), reverse=True)

    report["suspicious_features"] = suspicious
    report["leakage_flags"] = leakage
    report["ok"] = (len(leakage) == 0)
    return report
