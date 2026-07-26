"""
StockM v1.0 - Phase 6, Lesson 7
Time-Series Validation
======================

Why ordinary K-Fold is WRONG for stock prediction
-------------------------------------------------
Random K-Fold shuffles rows, so a fold can put 2025 data in the training
set and 2010 data in the validation set. The model then "trains on the
future to predict the past" - a fantasy that collapses the moment you
deploy (you cannot train on tomorrow). Random K-Fold leaks temporal
information and inflates every metric. For time-ordered data you MUST use
a validator that keeps training data strictly before validation data.

Three time-series validators (all leak-free)
-------------------------------------------
1. TimeSeriesSplit (sklearn, expanding train)
     train grows each fold; val is the next block forward.
     fold1: [----train----][val]
     fold2: [------train------][val]
     fold3: [--------train--------][val]
   Pros: uses more data over time; standard. Cons: train grows, so later
   folds are slower; assumes the past is always relevant (expanding window).

2. Expanding Window (custom, explicit boundaries)
     Same idea as TimeSeriesSplit but with an explicit boundary report so
   the logged experiment records exactly which date ranges trained vs
   validated each fold.

3. Rolling Window (custom, fixed train size)
     Train window of FIXED size slides forward; older data drops out.
     fold1: [train][val]
     fold2:    [train][val]
     fold3:       [train][val]
   Pros: model always trained on the most RECENT N rows - good when old
   regimes are no longer relevant (non-stationarity). Cons: less data per
   fold; discards history.

Which for StockM?
-----------------
Default = TimeSeriesSplit (expanding): more data helps with weak signal,
and our held-out test period (2023-2026) already enforces the ultimate
forward-only check. Rolling window is offered for robustness tests where
old regimes are suspected to hurt.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class CVSplitter(Protocol):
    """Any object yielding (train_idx, val_idx) with train strictly before val."""

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...
    def get_n_splits(self, X=None, y=None, groups=None) -> int: ...


def make_time_series_split(n_splits: int = 5) -> TimeSeriesSplit:
    """sklearn TimeSeriesSplit: expanding train, forward-only validation.

    Args:
        n_splits: Number of forward folds.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def expanding_window_indices(
    n_rows: int, n_splits: int = 5, val_size: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV: explicit boundary control.

    Each fold: train = [0 : k], val = [k : k + val_size]. Train grows.

    Args:
        n_rows:   Total rows available.
        n_splits: Number of folds.
        val_size: Rows per validation block; None = (n_rows // (n_splits + 1)).

    Returns:
        List of (train_idx, val_idx) numpy arrays.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if val_size is None:
        val_size = n_rows // (n_splits + 1)
    if val_size * n_splits + val_size > n_rows:
        raise ValueError(
            f"expanding window needs >= {(n_splits + 1) * val_size} rows, got {n_rows}"
        )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        train_end = k * val_size
        val_start = train_end
        val_end = train_end + val_size
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


def rolling_window_indices(
    n_rows: int, n_splits: int = 5, train_size: int | None = None,
    val_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling-window CV: fixed train size slides forward.

    Each fold: train = [k : k+train_size], val = [k+train_size : k+train_size+val_size].
    Older rows drop out as the window advances.

    Args:
        n_rows:     Total rows available.
        n_splits:   Number of folds.
        train_size: Fixed train window length; None = (n_rows // (n_splits + 2)).
        val_size:   Validation block length; None = same as train_size default budget.

    Returns:
        List of (train_idx, val_idx) numpy arrays.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if train_size is None:
        train_size = n_rows // (n_splits + 2)
    if val_size is None:
        val_size = max(1, n_rows // (n_splits + 2))
    if train_size + val_size + (n_splits - 1) * val_size > n_rows:
        raise ValueError("rolling window does not fit in n_rows")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        start = k * val_size
        train_idx = np.arange(start, start + train_size)
        val_idx = np.arange(start + train_size, start + train_size + val_size)
        folds.append((train_idx, val_idx))
    return folds


def describe_splits(
    splits: list[tuple[np.ndarray, np.ndarray]], dates
) -> list[dict[str, str]]:
    """Return a human-readable boundary report for each fold.

    Args:
        splits: List of (train_idx, val_idx).
        dates:  The DatetimeIndex (or index values) the indices refer to.

    Returns:
        List of {fold, train_range, val_range, train_n, val_n}.
    """
    import pandas as pd

    out: list[dict[str, str]] = []
    for i, (tr, va) in enumerate(splits, start=1):
        tr_dates = pd.Index(dates)[tr]
        va_dates = pd.Index(dates)[va]
        out.append({
            "fold": i,
            "train_range": f"{tr_dates.min()} -> {tr_dates.max()}",
            "val_range": f"{va_dates.min()} -> {va_dates.max()}",
            "train_n": int(len(tr)),
            "val_n": int(len(va)),
        })
    return out


def make_cv(method: str, n_splits: int = 5, **kwargs) -> CVSplitter:
    """Factory: pick a time-series validator by name.

    Args:
        method:    "timeseries" | "expanding" | "rolling".
        n_splits:  Number of folds.
        kwargs:    Extra args (e.g. train_size, val_size for rolling).

    Returns:
        A CVSplitter. For "timeseries" this is sklearn's TimeSeriesSplit
        (duck-types the Protocol); for the custom ones a small adapter.
    """
    method = method.lower()
    if method == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    if method in ("expanding", "expanding_window"):
        return _IndexListSplitter(
            lambda n: expanding_window_indices(n, n_splits=n_splits,
                                               val_size=kwargs.get("val_size"))
        )
    if method in ("rolling", "rolling_window"):
        return _IndexListSplitter(
            lambda n: rolling_window_indices(
                n, n_splits=n_splits,
                train_size=kwargs.get("train_size"),
                val_size=kwargs.get("val_size"),
            )
        )
    raise ValueError(f"Unknown CV method {method!r}")


class _IndexListSplitter:
    """Adapter so a list-of-index-pairs function duck-types sklearn's CV API."""

    def __init__(self, build_fn) -> None:
        self._build_fn = build_fn

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        folds = self._build_fn(len(X))
        for tr, va in folds:
            yield tr, va

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return 0
        return len(self._build_fn(len(X)))
