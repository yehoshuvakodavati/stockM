"""
StockM v1.0 - Phase 4, Lesson 6
Outlier Detection
=================

An outlier is a point far from the bulk of the distribution. In most domains
outliers are errors to remove; in finance they are often *the events that
matter* - the 2008 and 2020 crashes, earnings gap-ups, policy shocks. The
default stance is therefore RETAIN, but *understand*.

Two detection methods
---------------------
IQR method (robust, distribution-free):
    Outlier if  x < Q1 - 1.5*IQR  or  x > Q3 + 1.5*IQR.
    Based on quartiles, so immune to the very extremes it is measuring. The
    right choice for fat-tailed financial data.

Z-score method (assumes normality):
    Outlier if  |z| = |(x - mean)/std| > 3.
    Fast and intuitive, BUT mean and std are themselves distorted by the
    outliers, so z-scores are unreliable on heavy-tailed data. Use it as a
    cross-check, not the primary detector.

Should financial outliers be removed?
-------------------------------------
Usually NO. Removing crash days:
  - deletes the highest-signal rows (the model never learns tail risk),
  - biases training toward calm regimes (so it fails exactly when it
    matters most - in a crash).
Instead, robust preprocessing (RobustScaler, winsorising at extreme
percentiles, or log-transforming positively-skewed features) tames the
influence without deleting information. This module *detects and reports*;
removal is opt-in via a cap/floor (winsorise), never a drop, and only when
the user explicitly asks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_outliers(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    cols: list[str] | None = None,
) -> dict[str, Any]:
    """Report IQR + Z-score outlier counts per column. Does NOT mutate.

    Args:
        df:            Feature frame.
        z_threshold:   |z| above which a point is a Z-score outlier.
        iqr_multiplier: IQR fence multiplier (1.5 = Tukey's default).
        cols:          Restrict to these columns; default = all numeric.

    Returns:
        Dict keyed by column -> {iqr_outliers, z_outliers, pct_outliers}.
    """
    numeric = df.select_dtypes(include="number")
    if cols is not None:
        numeric = numeric[[c for c in cols if c in numeric.columns]]

    report: dict[str, Any] = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        n = len(s)
        if n < 4 or s.nunique() < 2:
            report[col] = {"iqr_outliers": 0, "z_outliers": 0, "pct_outliers": 0.0}
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low_fence = q1 - iqr_multiplier * iqr
        high_fence = q3 + iqr_multiplier * iqr
        iqr_out = int(((s < low_fence) | (s > high_fence)).sum())

        std = s.std()
        if std and not np.isnan(std) and std > 0:
            z = (s - s.mean()) / std
            z_out = int((z.abs() > z_threshold).sum())
        else:
            z_out = 0

        report[col] = {
            "iqr_outliers": iqr_out,
            "z_outliers": z_out,
            "pct_outliers": round(float(iqr_out) / float(n) * 100, 3),
        }
    return report


def winsorize(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Cap (not clip) values at the given quantiles - tames tails without
    deleting rows.

    Used only on the TRAIN split, then the same fence values are applied to
    val/test (the fences are train-derived to avoid leakage).

    Args:
        df:    Feature frame.
        lower: Lower quantile floor (0.01 = 1st percentile).
        upper: Upper quantile cap (0.99 = 99th percentile).
        cols:  Columns to winsorise; default = all numeric non-target.

    Returns:
        New DataFrame with capped values. Input not mutated.
    """
    out = df.copy()
    numeric = out.select_dtypes(include="number")
    if cols is not None:
        numeric = numeric[[c for c in cols if c in numeric.columns]]
    else:
        numeric = numeric[[c for c in numeric.columns if not c.startswith("target_")]]

    for col in numeric.columns:
        lo = numeric[col].quantile(lower)
        hi = numeric[col].quantile(upper)
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out
