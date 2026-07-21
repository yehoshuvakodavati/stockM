"""
StockM v1.0 - Phase 3, Lesson 4
Trend Features
==============

Moving averages are the simplest trend estimators: they smooth daily noise to
reveal the underlying direction.

- SMA (Simple Moving Average): unweighted mean of the last N closes. Each of
  the N days contributes equally. Reacts slowly; the larger N, the more lag.
- EMA (Exponential Moving Average): weighted mean where recent prices count
  more. Reacts faster than an SMA of the same window, so it tracks turns
  earlier at the cost of more noise.

Short vs long windows
---------------------
- Short windows (5, 10, 20) capture short-term trend / momentum.
- Long windows (50, 100, 200) capture the macro trend. The 200-day SMA is the
  classic institutional "bull vs bear market" line.

Lag effect
----------
A moving average is a *lagging* indicator: it summarises the past, so it
turns after price turns. We therefore also emit the *distance* of price from
each average (`price / sma - 1`), which is a leading, stationary signal:
"how extended are we vs the trend?" - the input a model actually learns from.

Leakage status: trailing windows only (min_periods set so no future data is
used). The EMA uses pandas' recursive formula with adjust=True, which depends
only on past values.
"""

from __future__ import annotations

import pandas as pd


def add_trend_features(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    sma_windows: tuple[int, ...] = (5, 10, 20, 50, 100, 200),
    ema_windows: tuple[int, ...] = (5, 10, 20, 50),
) -> pd.DataFrame:
    """Append SMA / EMA trend features and price-vs-average distances.

    Args:
        df:          OHLCV frame with DatetimeIndex.
        price_col:   Canonical adjusted price column.
        sma_windows: Window lengths for simple moving averages.
        ema_windows: Window lengths for exponential moving averages.

    Returns:
        New DataFrame with trend columns added. Input is not mutated.
    """
    out = df.copy()
    price = out[price_col]

    # --- Simple Moving Averages + distance-to-average ----------------------
    # min_periods=window: the SMA is NaN until N full days are available.
    # This is the honest representation - no partial-window guesswork that
    # could bias the early rows.
    for w in sma_windows:
        sma = price.rolling(window=w, min_periods=w).mean()
        out[f"sma_{w}"] = sma
        # Distance of price from its SMA, as a fraction. Stationary + leading.
        out[f"dist_sma_{w}"] = price / sma - 1.0

    # --- Exponential Moving Averages ---------------------------------------
    # span=N corresponds to an effective smoothing of ~2/(N+1). adjust=True is
    # the standard recursive EMA; it slightly weights the first observations
    # differently but never uses future data.
    for w in ema_windows:
        ema = price.ewm(span=w, adjust=True).mean()
        out[f"ema_{w}"] = ema
        out[f"dist_ema_{w}"] = price / ema - 1.0

    return out
