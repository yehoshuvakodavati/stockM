"""
StockM v1.0 - Phase 5, Lesson 3
Prepare Training Data
=====================

Loads the prepared train/validation/test CSVs (from Phase 4) and separates
them into feature matrices X and target vectors y.

The single most important rule here: **X and y must be physically separated,
and no target column may ever appear in X.** The targets are forward-looking
by construction (tomorrow's return); letting one leak into X is the #1 silent
model-killer in finance. This module enforces that contract with an explicit
assertion, not a convention.

Why features and targets stay separate
--------------------------------------
1. Leakage prevention - targets know the future; features must not.
2. Operational reality - at inference time only X is available; the code must
   never assume y is present when predicting.

The prepared data is already scaled (StandardScaler fit on train in Phase 4),
so X is model-ready for both linear (needs scaling) and tree (indifferent)
models. No rescaling happens here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"

TARGET_PREFIX = "target_"
SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}


def _safe_filename(symbol: str) -> str:
    return symbol.replace(".", "_")


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Return X columns = every numeric column that is NOT a target.

    This is the leakage firewall: anything starting with ``target_`` is
    excluded regardless of which target we predict, so even sibling targets
    (e.g. target_return_5d) cannot leak into a next-day-return model.
    """
    feats = [c for c in df.columns if not c.startswith(TARGET_PREFIX)]
    # Defensive: assert the chosen target is NOT in X.
    assert target_col not in feats, f"target {target_col} leaked into features!"
    return feats


def load_split(
    symbol: str, split: str, prepared_dir: Path | None = None
) -> pd.DataFrame:
    """Load one prepared split (train/validation/test) for a ticker.

    Args:
        symbol:      Ticker, e.g. "RELIANCE.NS".
        split:       "train" | "validation" | "test".
        prepared_dir: Override the default prepared directory.

    Returns:
        DataFrame with DatetimeIndex, sorted ascending.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {list(SPLIT_FILES)}, got {split!r}")
    base = prepared_dir or PREPARED_DIR
    path = base / _safe_filename(symbol) / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared {split} not found for {symbol}: {path}. "
            f"Run the EDA pipeline first (src/run_eda_pipeline.py)."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def split_xy(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared frame into (X, y).

    Args:
        df:         Prepared frame (features + targets).
        target_col: Column to use as y.

    Returns:
        (X, y) where X has all non-target columns and y is df[target_col].
    """
    if target_col not in df.columns:
        raise KeyError(
            f"target {target_col!r} not in prepared columns {list(df.columns)}"
        )
    feats = _feature_columns(df, target_col)
    X = df[feats].copy()
    y = df[target_col].copy()
    return X, y


def load_dataset(
    symbol: str,
    target_col: str = "target_next_return",
    prepared_dir: Path | None = None,
) -> dict[str, Any]:
    """Load the full train/val/test dataset for a ticker, split into X/y.

    Args:
        symbol:      Ticker, e.g. "RELIANCE.NS".
        target_col:  Regression target (default: next-day log return).
        prepared_dir: Override the default prepared directory.

    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, target_col, and the date ranges of each split.
    """
    train_df = load_split(symbol, "train", prepared_dir)
    val_df = load_split(symbol, "validation", prepared_dir)
    test_df = load_split(symbol, "test", prepared_dir)

    X_train, y_train = split_xy(train_df, target_col)
    X_val, y_val = split_xy(val_df, target_col)
    X_test, y_test = split_xy(test_df, target_col)

    # Verify the feature set is identical across splits (selection is fixed).
    feats = list(X_train.columns)
    assert list(X_val.columns) == feats, "feature mismatch train vs val"
    assert list(X_test.columns) == feats, "feature mismatch train vs test"

    # Linear models crash on NaN; trees/XGBoost tolerate it, but the prepared
    # data should already be clean. Drop any stray NaN rows defensively and
    # record the count so the loss is visible (never silent).
    def _drop_na(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        mask = X.notna().all(axis=1) & y.notna()
        return X[mask], y[mask]

    n_before = len(X_train)
    X_train, y_train = _drop_na(X_train, y_train)
    X_val, y_val = _drop_na(X_val, y_val)
    X_test, y_test = _drop_na(X_test, y_test)
    dropped = n_before - len(X_train)

    return {
        "symbol": symbol,
        "target_col": target_col,
        "feature_names": feats,
        "n_features": len(feats),
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "dropped_na_train": dropped,
        "date_ranges": {
            "train": (str(train_df.index.min().date()), str(train_df.index.max().date())),
            "val": (str(val_df.index.min().date()), str(val_df.index.max().date())),
            "test": (str(test_df.index.min().date()), str(test_df.index.max().date())),
        },
        "row_counts": {
            "train": len(X_train), "val": len(X_val), "test": len(X_test),
        },
    }
