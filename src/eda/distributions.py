"""
StockM v1.0 - Phase 4, Lesson 4
Distribution Analysis
=====================

Summary statistics hide shape. Two columns with identical mean and std can
have completely different distributions - one bell-shaped, one with a spike
and a long tail. Distribution analysis recovers the shape.

Shape metrics
-------------
skewness    Asymmetry.
              skew ~ 0  -> symmetric
              skew > 0  -> right tail (a few large positive values; e.g. raw
                           returns have mild positive skew from gap-ups)
              skew < 0  -> left tail (a few large negative values; crash days
                           make return distributions often *negatively* skewed
                           at longer horizons)
kurtosis    Tail weight relative to a normal distribution.
              ~3 (Fisher 0)  -> normal-like tails
              > 3 (Fisher >0) -> FAT tails: extreme events more likely than
                                normal. Stock returns typically show kurtosis
                                of 5-20+ - this is the single most important
                                distributional fact in finance.

Why distributions affect ML models
----------------------------------
- Linear models and neural nets assume-ish normality; fat tails produce
  extreme inputs that dominate gradients / least-squares loss.
- Tree models are rank-based and care less, but fat tails still mean rare
  regimes the model sees too few times to learn.
- Scaling choice (Lesson 10) depends on shape: heavy tails + outliers favour
  RobustScaler over StandardScaler.

Normality test
--------------
We use D'Agostino's K^2 (scipy) where available - a combined skew+kurtosis
test. p < 0.05 => reject normality. For ~5,000 rows almost any financial
column will reject normality; the test is most useful as a sanity flag for
columns that *don't* reject (suspiciously normal => maybe synthetic/constant).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:  # scipy is in requirements; guard anyway for minimal environments.
    from scipy import stats as sp_stats
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def distribution_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute skewness, (excess) kurtosis, and a normality flag per column.

    Args:
        df: Feature frame.

    Returns:
        Dict keyed by column -> {skew, kurtosis, is_normal, normality_p}.
    """
    numeric = df.select_dtypes(include="number")
    result: dict[str, Any] = {}

    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s) < 8 or s.nunique() < 2:
            # Constant / near-constant columns have undefined moments.
            result[col] = {
                "skew": None, "kurtosis": None,
                "is_normal": False, "normality_p": None,
            }
            continue

        skew = float(s.skew())
        kurt = float(s.kurtosis())  # pandas returns *excess* kurtosis (Fisher)

        if _HAVE_SCIPY and len(s) >= 20:
            try:
                _, p = sp_stats.normaltest(s.to_numpy())
                is_normal = bool(p > 0.05)
                p = float(p)
            except Exception:  # pragma: no cover
                p, is_normal = None, False
        else:
            p, is_normal = None, False

        result[col] = {
            "skew": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "is_normal": is_normal,
            "normality_p": round(p, 6) if p is not None else None,
        }
    return result
