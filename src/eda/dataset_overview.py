"""
StockM v1.0 - Phase 4, Lesson 2
Dataset Overview
================

The first thing a senior quant does with a new dataset is *look at its
skeleton*: how big is it, what types are the columns, where are the holes,
are there duplicates, how much memory does it eat. Before any statistics,
this structural pass catches the gross defects (a column that silently
became `object`, duplicate rows that would leak, a 12 GB frame that won't fit
in memory for training).

This module produces a structured overview dict - no mutation, no plotting -
so it can be logged and diffed across runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return a structured overview of shape, dtypes, missing, dupes, memory.

    Args:
        df: Engineered feature frame (features + targets), DatetimeIndex.

    Returns:
        Dict with shape, dtypes summary, missing-value summary, duplicate
        counts, and memory usage.
    """
    overview: dict[str, Any] = {}

    # --- Shape -------------------------------------------------------------
    overview["rows"] = int(df.shape[0])
    overview["columns"] = int(df.shape[1])
    overview["date_range"] = (
        str(df.index.min().date()), str(df.index.max().date())
    ) if isinstance(df.index, pd.DatetimeIndex) else None
    overview["index_type"] = type(df.index).__name__

    # --- Data types --------------------------------------------------------
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    overview["dtype_counts"] = {k: int(v) for k, v in dtype_counts.items()}
    # Non-numeric columns are a red flag for a feature matrix.
    non_numeric = [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    overview["non_numeric_columns"] = non_numeric

    # --- Missing values ----------------------------------------------------
    na_per_col = df.isna().sum()
    overview["missing"] = {
        "total_cells": int(df.size),
        "total_missing": int(na_per_col.sum()),
        "pct_missing": round(float(na_per_col.sum()) / float(df.size) * 100, 4)
        if df.size else 0.0,
        "columns_with_missing": int((na_per_col > 0).sum()),
    }

    # --- Duplicate records -------------------------------------------------
    # Duplicate *rows* (same values) - a leak risk if train/test overlap.
    # Duplicate *index* (same date) - a pipeline bug.
    overview["duplicates"] = {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_index": int(df.index.duplicated().sum()),
    }

    # --- Memory usage ------------------------------------------------------
    # deep=True introspects object columns; useful to spot bloated dtypes.
    overview["memory_mb"] = round(float(df.memory_usage(deep=True).sum() / 1e6), 3)

    return overview
