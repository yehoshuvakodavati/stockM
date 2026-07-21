"""
StockM v1.0 - Phase 3, Lesson 9
Lag Features
============

A model sees one row at a time. Without lag features, that row is
memoryless - it cannot know what happened yesterday or last week. Lag
features give the model explicit *historical context*: "where was price /
volume / return 1, 3, 5, 10, 20 days ago?"

Lags are the foundation of time-series forecasting. Every autoregressive
model (AR, ARIMA) and most sequence models are, at heart, learning from
lagged values. Providing them as explicit columns lets even a non-sequential
model (XGBoost, MLP) exploit temporal structure.

What we lag
-----------
- Close (adjusted): prior price levels.
- Open: prior opening levels.
- Volume: prior participation.
- Returns: prior daily returns (momentum/mean-reversion signal).

Periods: 1, 3, 5, 10, 20 trading days - covering intraweek to ~monthly memory.

Leakage status: lags use shift(N) with N>=1 - they reference strictly past
rows. A positive shift would be future leakage; we never use it for inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    lag_periods: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> pd.DataFrame:
    """Append lagged close / open / volume / return columns.

    Args:
        df:          OHLCV frame with DatetimeIndex.
        price_col:   Canonical adjusted price (lagged as `close_lag_N`).
        lag_periods: Number of days to shift back.

    Returns:
        New DataFrame with lag columns added. Input not mutated.
    """
    out = df.copy()

    price = out[price_col]
    open_ = out["Open"]
    volume = out["Volume"]

    # Daily return series recomputed locally so this module is self-contained.
    log_ret = np.log(price / price.shift(1))

    for n in lag_periods:
        out[f"close_lag_{n}"] = price.shift(n)
        out[f"open_lag_{n}"] = open_.shift(n)
        out[f"volume_lag_{n}"] = volume.shift(n)
        out[f"return_lag_{n}"] = log_ret.shift(n)

    return out
