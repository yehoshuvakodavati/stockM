"""
StockM v1.0 - Phase 3, Lesson 8
Time Features
=============

Financial markets are not time-invariant: they exhibit calendar regularities
(seasonality) that a model can exploit if it knows *when* it is.

Components extracted
--------------------
year, month, quarter, week, day_of_week, day_of_month, trading_day_number.

- day_of_week (Mon=0 .. Sun=6): captures the "Monday effect" / weekend effect
  and weekday-specific flow patterns.
- month / quarter: captures month-of-year effects such as the January effect,
  quarter-end rebalancing, and earnings-season clustering.
- trading_day_number: a monotonically increasing ordinal for the ticker. Gives
  the model a sense of "how far along this series is" - useful for detecting
  regime maturity and for trend features that implicitly depend on history
  length.

Cyclic encoding note
--------------------
Raw month/day numbers are *ordinal*, not metric: December (12) is not "twice
June (6)", and January (1) is adjacent to December, not far from it. Linear
models mis-handle this. For tree models, raw integers are fine (trees split
on thresholds). For linear/neural models, a sin/cos cyclical encoding is
correct - we emit both the raw integer (for trees) and the cyclic pair.

Leakage status: time components are deterministic properties of the date
itself - no future market data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append calendar / ordinal time features derived from the DatetimeIndex.

    Args:
        df: OHLCV frame with a DatetimeIndex.

    Returns:
        New DataFrame with time columns added. Input not mutated.
    """
    out = df.copy()
    idx = out.index

    # Guard: caller may have passed a RangeIndex by mistake.
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(
            f"time_features require a DatetimeIndex, got {type(idx).__name__}"
        )

    # --- Raw calendar components (good for tree models) --------------------
    out["year"] = idx.year
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["week"] = idx.isocalendar().week.astype("int64").to_numpy()
    out["day_of_week"] = idx.dayofweek                      # Mon=0 .. Sun=6
    out["day_of_month"] = idx.day

    # --- Cyclic encoding (good for linear / neural models) -----------------
    # sin/cos map the cycle onto a circle so Jan and Dec are neighbours.
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    out["day_of_week_sin"] = np.sin(2.0 * np.pi * out["day_of_week"] / 7.0)
    out["day_of_week_cos"] = np.cos(2.0 * np.pi * out["day_of_week"] / 7.0)

    # --- Trading-day ordinal ----------------------------------------------
    # 1-based position of each row within this ticker's series. Cheap
    # regime-maturity signal; also useful as a join key / debugging aid.
    out["trading_day_number"] = np.arange(1, len(out) + 1)

    return out
