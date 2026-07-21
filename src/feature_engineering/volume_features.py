"""
StockM v1.0 - Phase 3, Lesson 7
Volume Features
===============

Price tells you *what* happened; volume tells you *how much conviction* was
behind it. A 3% up-move on twice-normal volume is a very different signal
from a 3% up-move on half-normal volume. Analysing price without volume is
reading half the sentence.

Indicators
----------
Volume change:
    Volume vs prior day - detects sudden participation spikes.

Volume moving average (VMA):
    Trailing mean of volume over N days - the "normal" level for context.

Volume ratio:
    Volume / VMA. >1 = above-average participation; <1 = quiet. This is the
    single most useful volume feature: it is stationary and cross-ticker
    comparable (unlike raw share counts, which differ wildly by float).

On-Balance Volume (OBV):
    A cumulative running total: add today's volume if close > prev close,
    subtract if lower, unchanged if equal. OBV translates volume into a
    price-direction-weighted pressure series. Divergences between OBV and
    price (e.g. price rising but OBV falling) often precede reversals.

Volume momentum:
    VMA ratio of recent volume vs older volume - is participation accelerating
    or decaying?

Leakage status: all features use trailing windows and shift(1). OBV is
cumulative over the past only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume. Cumulative; depends only on past + current row.

    sign(C_t - C_{t-1}) * V_t, cumulatively summed. The shift(1) inside the
    comparison guarantees no look-ahead.
    """
    direction = np.sign(close.diff())
    # On no-change days direction is 0 -> contributes 0, which is correct.
    return (direction * volume).cumsum()


def add_volume_features(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    vma_windows: tuple[int, ...] = (5, 10, 20),
    vol_momentum_fast: int = 5,
    vol_momentum_slow: int = 20,
) -> pd.DataFrame:
    """Append volume change, VMA, volume ratio, OBV, volume momentum.

    Args:
        df:                  OHLCV frame with DatetimeIndex.
        price_col:           Price column used for OBV direction.
        vma_windows:         Window lengths for volume moving averages / ratios.
        vol_momentum_fast:   Recent-volume window for momentum ratio.
        vol_momentum_slow:   Baseline window for momentum ratio.

    Returns:
        New DataFrame with volume columns added. Input not mutated.
    """
    out = df.copy()
    close = out[price_col]
    volume = out["Volume"]

    # --- Volume change (raw + ratio) ---------------------------------------
    out["volume_change"] = volume.diff()
    # Ratio vs prior day, guarded against zero volume.
    prev_vol = volume.shift(1).replace(0.0, np.nan)
    out["volume_change_ratio"] = volume / prev_vol

    # --- VMA + volume ratio ------------------------------------------------
    for w in vma_windows:
        vma = volume.rolling(window=w, min_periods=w).mean()
        out[f"vma_{w}"] = vma
        out[f"volume_ratio_{w}"] = volume / vma.replace(0.0, np.nan)

    # --- On-Balance Volume -------------------------------------------------
    out["obv"] = _obv(close, volume)

    # --- Volume momentum (acceleration of participation) -------------------
    fast = volume.rolling(window=vol_momentum_fast, min_periods=vol_momentum_fast).mean()
    slow = volume.rolling(window=vol_momentum_slow, min_periods=vol_momentum_slow).mean()
    out["volume_momentum"] = fast / slow.replace(0.0, np.nan)

    return out
