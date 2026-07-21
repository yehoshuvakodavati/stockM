"""
StockM v1.0 - Phase 4, Lesson 3
Statistical Analysis
====================

Descriptive statistics summarise a column's central tendency, spread, and
extremes in a few numbers. They are the first quantitative read on "what
does this feature look like".

What each statistic tells us about stock data
---------------------------------------------
mean        Average. For *prices* it is misleading (non-stationary, dragged by
            trend). For *returns* it is the drift - usually a tiny positive
            number (equities drift up).
median      The middle value. Robust to outliers, so for fat-tailed return
            columns it is the honest "typical day". mean >> median => right
            skew (a few big up-days).
mode        Most frequent value. Rarely useful for continuous features, but
            relevant for discrete targets (the modal class is the majority
            class - the null-model prediction).
std         Dispersion. For returns this is *volatility* - the core risk
            number. Annualised x sqrt(252) it becomes "the stock's vol".
var         std^2. Used by variance-threshold feature selection.
min / max   The observed range. For RSI these must be ~0 and ~100; a min of
            -inf or max of inf flags a divide-by-zero bug.
quartiles   Q1/median/Q3 give the interquartile range (IQR), the basis of
            outlier detection (Lesson 6).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def descriptive_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute extended descriptive statistics for every numeric column.

    Wraps ``df.describe`` with mode and the IQR, and returns a tidy dict.

    Args:
        df: Feature frame (numeric columns are analysed; others skipped).

    Returns:
        Dict keyed by column -> {mean, median, std, min, max, q1, q3, iqr, ...}.
    """
    numeric = df.select_dtypes(include="number")

    # Pandas describe gives count/mean/std/min/25%/50%/75%/max in one shot.
    desc = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T

    stats: dict[str, Any] = {}
    for col, row in desc.iterrows():
        # Mode: take the first modal value only (multimodal is rare here).
        mode_val = numeric[col].mode(dropna=True)
        mode_v = float(mode_val.iloc[0]) if len(mode_val) else None

        q1, q3 = float(row["25%"]), float(row["75%"])
        stats[col] = {
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "median": float(row["50%"]),
            "mode": mode_v,
            "std": float(row["std"]),
            "var": float(row["std"] ** 2) if not np.isnan(row["std"]) else None,
            "min": float(row["min"]),
            "max": float(row["max"]),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
        }
    return stats
