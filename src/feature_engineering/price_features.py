"""
StockM v1.0 - Phase 3, Lesson 3
Price-Based Features
====================

Every feature here is derived from the OHLCV price columns. They are the most
fundamental transformations - the atoms from which trend/momentum/volatility
features are later built.

Canonical price series
----------------------
We use ``Adj Close`` as the canonical price (`price`) for return and trend
math because it is retroactively adjusted for splits and dividends. Using raw
``Close`` would inject artificial jumps on ex-split/ex-dividend days that the
model would mistake for real signal. The structural OHLC features
(high-low range, open-close gap) are scale-invariant ratios, so raw OHLC is
safe there: a split divides H/L/O/C by the same factor and the ratio is
unchanged.

Leakage status: all features use only the current and prior rows
(``shift(1)``). None look forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Append price-based features to a copy of ``df``.

    Args:
        df:        OHLCV frame with a DatetimeIndex and columns including
                   ``Adj Close``, ``Close``, ``High``, ``Low``, ``Open``.
        price_col: Column to treat as the canonical (adjusted) price.

    Returns:
        A new DataFrame with the original columns plus the price features.
        The input is not mutated.
    """
    out = df.copy()

    price = out[price_col]
    close = out["Close"]
    open_ = out["Open"]
    high = out["High"]
    low = out["Low"]

    # --- Returns -----------------------------------------------------------
    # Daily return: fractional change vs prior close. Stationary (unlike price).
    out["daily_return"] = price.pct_change()

    # Percentage change over 1 day on the adjusted price (same value as
    # daily_return; kept under its roadmap name for clarity / discoverability).
    out["pct_change_1d"] = price.pct_change(periods=1)

    # Log return: ln(P_t / P_{t-1}). Preferred in quant finance because log
    # returns are additive over time (sum of daily log returns = multi-period
    # log return), which simplifies compounding and volatility math.
    out["log_return"] = np.log(price / price.shift(1))

    # --- Price differences (expressed as fractions for scale invariance) ---
    # Absolute rupee differences are non-stationary across splits/price levels,
    # so we store the fractional form. The raw difference is recoverable as
    # price * fraction if ever needed.
    out["price_diff"] = price.diff()                 # absolute (rupees) - reference
    out["price_diff_pct"] = price.diff() / price.shift(1)

    # --- Intraday structure ------------------------------------------------
    # High-Low range as a fraction of close: intraday volatility proxy.
    # Scale-invariant (split-safe), so raw H/L/C are fine here.
    out["high_low_range"] = (high - low) / close

    # Open-Close gap as a fraction of open: intraday conviction/direction.
    out["open_close_diff"] = (close - open_) / open_

    # --- Composite prices --------------------------------------------------
    # Typical price: the day's average trading level (H+L+C)/3. Used as the
    # basis for money-flow and many volume-weighted indicators.
    out["typical_price"] = (high + low + close) / 3.0

    # Weighted close: (H+L+2C)/4. Weights the close twice, emphasising the
    # most informative of the four prices.
    out["weighted_close"] = (high + low + 2.0 * close) / 4.0

    return out
