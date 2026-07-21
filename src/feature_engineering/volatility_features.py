"""
StockM v1.0 - Phase 3, Lesson 6
Volatility Features
===================

Volatility = the dispersion of returns. It is the primary measure of risk and
the regime variable that governs position sizing, stop placement, and option
pricing. Crucially, volatility *clusters*: high-vol days follow high-vol days
(GARCH effect), so recent volatility is itself predictive of near-future
volatility.

Indicators
----------
Rolling standard deviation of returns (default 20):
    The most direct volatility measure - sigma of daily log returns over a
    trailing window. Annualised x sqrt(252) for comparability.

ATR (Average True Range, default 14):
    True Range = max(H-L, |H-prev C|, |L-prev C|). Captures intraday range
    *and* overnight gaps, which plain H-L misses. ATR is the TR moving
    average. The canonical risk-per-trade unit in trading systems.

Bollinger Bands (default 20, 2 sigma):
    sma +/- k*std. Price near the upper band = extended up; near lower =
    extended down. The bands self-widen in volatile regimes and narrow in
    quiet ones - a regime detector.

Bollinger width:
    (upper - lower) / sma. A stationary, normalised measure of how wide the
    bands are - i.e. current volatility regime relative to price level.

Historical volatility (default 21):
    annualised std of log returns over N days. The number quoted as "the
    stock's volatility" in finance (e.g. "20% vol").

Leakage status: trailing windows + shift(1) for TR's prev-close term only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # approximate NSE/NYSE trading days per year


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H - prev_close|, |L - prev_close|).

    ``prev_close`` uses shift(1) - strictly the prior day, never the next.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def add_volatility_features(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    vol_window: int = 20,
    atr_window: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    hv_window: int = 21,
) -> pd.DataFrame:
    """Append rolling std, ATR, Bollinger bands, Bollinger width, hist vol.

    Args:
        df:        OHLCV frame with DatetimeIndex.
        price_col: Canonical adjusted price column.
        vol_window:  Window for rolling return std.
        atr_window:  Window for ATR smoothing.
        bb_window:   Bollinger SMA window.
        bb_std:      Bollinger band width in std devs.
        hv_window:   Historical-volatility window (annualised).

    Returns:
        New DataFrame with volatility columns added. Input not mutated.
    """
    out = df.copy()
    price = out[price_col]
    high, low, close = out["High"], out["Low"], out["Close"]

    # Need log returns as the volatility basis. Recompute locally so this
    # module is self-contained (does not depend on price_features ordering).
    log_ret = np.log(price / price.shift(1))

    # --- Rolling std of returns (raw + annualised) -------------------------
    roll_std = log_ret.rolling(window=vol_window, min_periods=vol_window).std()
    out[f"vol_std_{vol_window}"] = roll_std
    out[f"vol_ann_{vol_window}"] = roll_std * np.sqrt(TRADING_DAYS)

    # --- ATR ----------------------------------------------------------------
    tr = _true_range(high, low, close)
    out["atr"] = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    # ATR as a fraction of price - stationary and cross-ticker comparable.
    out["atr_pct"] = out["atr"] / close

    # --- Bollinger Bands ---------------------------------------------------
    bb_sma = price.rolling(window=bb_window, min_periods=bb_window).mean()
    bb_sd = price.rolling(window=bb_window, min_periods=bb_window).std()
    out["bb_upper"] = bb_sma + bb_std * bb_sd
    out["bb_lower"] = bb_sma - bb_std * bb_sd
    out["bb_mid"] = bb_sma
    # Width normalised by the mid band -> scale-free regime indicator.
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_sma
    # Position of price within the bands in [0,1]-ish range: leading signal.
    out["bb_pct"] = (price - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

    # --- Historical volatility (annualised) --------------------------------
    out[f"hist_vol_{hv_window}"] = (
        log_ret.rolling(window=hv_window, min_periods=hv_window).std()
        * np.sqrt(TRADING_DAYS)
    )

    return out
