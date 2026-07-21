"""
StockM v1.0 - Phase 3, Lesson 5
Momentum Features
=================

Momentum measures the *speed* and *strength* of price moves - not the
direction alone. Trend tells you "up"; momentum tells you "accelerating up,
exhausted, or about to reverse".

Indicators
----------
RSI (Relative Strength Index, default 14):
    Maps the ratio of average up-move to average down-move into a 0-100 band.
    - >70 conventionally "overbought" (extended, reversal risk)
    - <30 "oversold" (beaten down, bounce risk)
    RSI is bounded and stationary - ideal as a model input.

MACD (Moving Average Convergence Divergence):
    ema_fast(12) - ema_slow(26). Positive => short trend above long trend.
    - Signal line: ema(9) of MACD. Crossovers of MACD over its signal are the
      classic buy/sell triggers.
    - Histogram: MACD - Signal. The *rate of change* of the trend strength;
      histogram turning positive/negative is an early momentum shift.

ROC (Rate of Change):
    (P_t / P_{t-N}) - 1. Pure N-period return - momentum in its rawest form.

Momentum indicator:
    P_t - P_{t-N} (absolute). Kept for completeness; prefer ROC (stationary).

Market psychology
-----------------
Momentum exists because information diffuses slowly and investors
under-react, then over-react. A trend therefore tends to persist (momentum)
before snapping back (mean reversion) - both are exploitable signals.

Leakage status: all indicators use trailing windows / shifts only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. Pure past data via rolling + shift.

    Uses the simple-running-mean variant (avg of gains / avg of losses over
    the window) rather than Wilder's recursive smoothing; both are
    leak-free, and the running-mean form is easier to reason about and
    matches most library defaults.

    Implementation note: we stay in float64 throughout and avoid injecting
    ``pd.NA`` into the computation. Mixing nullable-float (``Float64``) with
    Python floats via ``replace(0.0, pd.NA)`` can degrade the result to
    ``object`` dtype, which would silently break distance-based / neural
    models downstream. The no-loss (divide-by-zero) case is handled with
    ``np.where`` instead.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0)).astype("float64")

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Guard divide-by-zero with np.where so the column stays float64:
    #   avg_loss == 0  -> no down-days in window -> RSI = 100 (overbought)
    #   else           -> RSI = 100 - 100/(1 + avg_gain/avg_loss)
    avg_loss_arr = avg_loss.to_numpy()
    rs_arr = np.divide(
        avg_gain.to_numpy(),
        avg_loss_arr,
        out=np.full_like(avg_gain.to_numpy(), np.nan, dtype="float64"),
        where=avg_loss_arr != 0.0,
    )
    rsi_arr = np.where(
        (avg_loss_arr == 0.0) & ~np.isnan(avg_gain.to_numpy()),
        100.0,
        100.0 - (100.0 / (1.0 + rs_arr)),
    )
    return pd.Series(rsi_arr, index=series.index, name="rsi", dtype="float64")


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram). Trailing EMAs only."""
    ema_fast = series.ewm(span=fast, adjust=True).mean()
    ema_slow = series.ewm(span=slow, adjust=True).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=True).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_momentum_features(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    roc_periods: tuple[int, ...] = (5, 10, 20),
    momentum_periods: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Append RSI, MACD family, ROC and momentum indicators.

    Args:
        df:               OHLCV frame with DatetimeIndex.
        price_col:        Canonical adjusted price column.
        rsi_window:       RSI lookback (industry standard 14).
        macd_fast/slow/signal: MACD EMA spans.
        roc_periods:      N for (P_t/P_{t-N})-1.
        momentum_periods: N for P_t - P_{t-N}.

    Returns:
        New DataFrame with momentum columns added. Input not mutated.
    """
    out = df.copy()
    price = out[price_col]

    # --- RSI ---------------------------------------------------------------
    out["rsi"] = _rsi(price, window=rsi_window)

    # --- MACD family -------------------------------------------------------
    macd_line, signal_line, hist = _macd(
        price, fast=macd_fast, slow=macd_slow, signal=macd_signal
    )
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # --- Rate of Change ----------------------------------------------------
    for n in roc_periods:
        out[f"roc_{n}"] = price / price.shift(n) - 1.0

    # --- Momentum (absolute) ----------------------------------------------
    for n in momentum_periods:
        out[f"momentum_{n}"] = price - price.shift(n)

    return out
