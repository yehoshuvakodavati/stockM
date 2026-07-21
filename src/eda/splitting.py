"""
StockM v1.0 - Phase 4, Lesson 11
Dataset Splitting
=================

Time-series data MUST be split chronologically, never randomly. Random
shuffling puts future rows in the training set and past rows in the test
set - the model trains on information it would not have at prediction time
(the definition of leakage) and backtest numbers become fantasy.

Chronological split
-------------------
Given a time-sorted frame, cut it into contiguous blocks:

    |-------- train --------||- gap -|-- val --||- gap -|-- test --|
                past                                          future

The ``gap`` between blocks is critical: it prevents *label leakage* across
the boundary. Our target is next-day return, so a row at the very end of
train has a label that looks 1 day into the val period. A gap of
``horizon_days`` (here 1, matching data_config.yaml) guarantees no train
row's label overlaps any val/test row.

Splits are ratios (e.g. 0.70 / 0.15 / 0.15). We honour data_config.yaml's
``test_size`` and ``val_size`` and ``gap_days``.

No shuffling, ever.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    gap_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split a time-sorted frame into train / val / test with gaps.

    Args:
        df:        Feature frame with a DatetimeIndex, sorted ascending.
        val_size:  Fraction of rows for validation.
        test_size: Fraction of rows for test.
        gap_days:  No-leakage gap (in rows) between train/val and val/test.
                   Must be >= the target horizon so labels don't overlap.

    Returns:
        (train, val, test, info) where info records the cut indices/counts.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("chronological_split requires a DatetimeIndex.")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    df = df.sort_index()
    n = len(df)

    # Compute cut points from the END (test last, then val, train keeps rest).
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    gap = max(int(gap_days), 0)

    # test = last n_test rows
    test_start = n - n_test
    # val = block before test, separated by gap
    val_end = test_start - gap
    val_start = val_end - n_val
    # train = everything before val, separated by gap
    train_end = val_start - gap

    train = df.iloc[:train_end]
    val = df.iloc[val_start:val_end]
    test = df.iloc[test_start:]

    info = {
        "total_rows": int(n),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_end": str(train.index.max().date()) if len(train) else None,
        "val_start": str(val.index.min().date()) if len(val) else None,
        "val_end": str(val.index.max().date()) if len(val) else None,
        "test_start": str(test.index.min().date()) if len(test) else None,
        "test_end": str(test.index.max().date()) if len(test) else None,
        "gap_days": gap,
        "method": "chronological",
    }
    return train, val, test, info
