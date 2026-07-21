"""
StockM v1.0 - Phase 3, Lesson 13
Feature Validation
==================

After every feature family has run, the resulting frame is *audited* before
it is saved. This is the second line of defence (the first being each
family's own leak-free construction). The validator does not mutate the
frame; it returns a structured report so the orchestrator can log it and
(optionally) refuse to save a bad dataset.

Checks
------
1. Missing values     - per-column NaN counts (warmup rows are expected;
                        a spike mid-series is a bug).
2. Infinite values    - +inf / -inf from divide-by-zero in ratio features.
3. Duplicate columns  - two columns with the same name (silent bug source).
4. Constant features  - zero variance => no information, pure noise.
5. Correlation        - pairs of features with |corr| above a threshold
                        (multicollinearity). Reported, not removed - removal
                        is the job of feature selection (Session 9).
6. Data types         - non-numeric columns that would break a model.

The report is a plain dict so it can be logged as JSON and diffed across runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Columns that are legitimately forward-looking and must never be used as
# model inputs. The validator uses this to exclude targets from the
# multicollinearity/constant checks (a target can be anything).
TARGET_PREFIX = "target_"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric, non-target column names - the model-input candidates."""
    # include="number" covers BOTH numpy numeric dtypes and pandas nullable
    # extension dtypes (Int64 etc.); targets are excluded by name regardless.
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric if not c.startswith(TARGET_PREFIX)]


def validate_features(
    df: pd.DataFrame,
    corr_threshold: float = 0.95,
    n_correlated_report: int = 20,
) -> dict[str, Any]:
    """Run the full validation suite and return a structured report.

    Args:
        df:                   The engineered frame (features + targets).
        corr_threshold:       |Pearson r| above which a pair is flagged.
        n_correlated_report:  Max correlated pairs to list in the report.

    Returns:
        Dict with keys: missing, infinite, duplicate_columns, constant,
        correlated, dtype_issues, feature_count, row_count, ok.
    """
    report: dict[str, Any] = {}

    report["row_count"] = int(len(df))
    report["column_count"] = int(df.shape[1])

    # 1. Missing values -----------------------------------------------------
    na = df.isna().sum()
    report["missing"] = {
        "total_cells": int(df.size),
        "total_na": int(na.sum()),
        "pct_na": round(float(na.sum()) / float(df.size) * 100.0, 3) if df.size else 0.0,
        "columns_with_na": int((na > 0).sum()),
        # Top offenders only, to keep the report readable.
        "top_columns": {
            c: int(v) for c, v in na[na > 0].sort_values(ascending=False).head(10).items()
        },
    }

    # 2. Infinite values ----------------------------------------------------
    # inf can only live in floating columns (Int64 cannot represent it), so
    # restrict to floats - this also avoids np.isinf on nullable Int64 which
    # would otherwise raise.
    float_cols = df.select_dtypes(include=[np.floating])
    inf = float_cols.apply(lambda s: int(np.isinf(s.to_numpy()).sum()))
    inf = inf[inf > 0]
    report["infinite"] = {
        "columns_with_inf": int(len(inf)),
        "top_columns": {c: int(v) for c, v in inf.head(10).items()},
    }

    # 3. Duplicate columns --------------------------------------------------
    dupes = df.columns[df.columns.duplicated()].tolist()
    report["duplicate_columns"] = dupes

    # 4. Constant features (zero variance) ---------------------------------
    feats = _feature_columns(df)
    if feats:
        variances = df[feats].var(skipna=True, numeric_only=True)
        constant = variances[variances == 0].index.tolist()
    else:
        constant = []
    report["constant"] = constant

    # 5. Correlation / multicollinearity -----------------------------------
    correlated: list[dict[str, Any]] = []
    if len(feats) >= 2:
        # Drop NA-pairs efficiently via .corr() (pairwise complete).
        corr = df[feats].corr(method="pearson").abs()
        # Keep only the upper triangle to avoid duplicate (a,b) & (b,a).
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        stacked = upper.stack()
        high = stacked[stacked >= corr_threshold].sort_values(ascending=False)
        for (a, b), val in high.head(n_correlated_report).items():
            correlated.append({"a": a, "b": b, "corr": round(float(val), 4)})
    report["correlated"] = correlated
    report["correlated_pair_count"] = int(len(correlated))

    # 6. Data types ---------------------------------------------------------
    # Use pandas' dtype check (handles numpy + nullable extension dtypes);
    # np.issubdtype crashes on pandas Int64Dtype.
    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and c not in df.index.names
    ]
    report["dtype_issues"] = non_numeric

    # 7. Headline verdict ---------------------------------------------------
    report["feature_count"] = int(len(feats))
    report["target_count"] = int(
        sum(1 for c in df.columns if c.startswith(TARGET_PREFIX))
    )
    # "ok" = no hard blockers. Missing/inf at the warmup boundary are expected
    # and handled by the orchestrator's dropna; duplicates/constants are bugs.
    report["ok"] = (not dupes) and (not constant) and (not non_numeric)

    return report
