"""
StockM v1.0 - Phase 3, Lesson 10
Rolling Window Features
=======================

A rolling window computes a statistic over the last N rows, then slides
forward one row and recomputes. It is the engine behind moving averages,
rolling volatility, and any "recent behaviour" summary.

Unlike a single lag (one past point), a rolling statistic summarises a
*window* of past points - it is smoother and more robust to single-day noise.

Statistics emitted (window W, default 20)
-----------------------------------------
- rolling_mean_W:    recent average level (trend context)
- rolling_max_W:     recent high - resistance / breakout reference
- rolling_min_W:     recent low  - support / breakdown reference
- rolling_median_W:  robust central tendency (resistant to outliers vs mean)
- rolling_var_W:     dispersion of price over the window
- rolling_vol_W:     std of log returns over the window (annualised) - risk

min_periods
-----------
We set min_periods=window so each statistic is only emitted once a full
window is available. Partial-window statistics would be biased (a 20-day
mean computed from 3 days is not a 20-day mean) and would inject unstable
values into the early rows. The cost is NaN warmup rows, which the pipeline
drops later (Lesson 12).

Leakage status: trailing windows only - the window ends at the current row
and looks backward. No centered windows (those would look forward).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def add_rolling_features(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Append rolling mean/max/min/median/variance/volatility per window.

    Args:
        df:        OHLCV frame with DatetimeIndex.
        price_col: Canonical adjusted price column.
        windows:   Trailing window lengths.

    Returns:
        New DataFrame with rolling columns added. Input not mutated.
    """
    out = df.copy()
    price = out[price_col]
    log_ret = np.log(price / price.shift(1))

    for w in windows:
        roller = price.rolling(window=w, min_periods=w)
        out[f"rolling_mean_{w}"] = roller.mean()
        out[f"rolling_max_{w}"] = roller.max()
        out[f"rolling_min_{w}"] = roller.min()
        out[f"rolling_median_{w}"] = roller.median()
        out[f"rolling_var_{w}"] = roller.var()

        # Rolling volatility on *returns*, not price - price variance is
        # non-stationary across levels. Annualised for comparability.
        out[f"rolling_vol_{w}"] = (
            log_ret.rolling(window=w, min_periods=w).std() * np.sqrt(TRADING_DAYS)
        )

    return out
