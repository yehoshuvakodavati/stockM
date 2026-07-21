"""
StockM v1.0 - Phase 3, Lesson 11
Target Variable Engineering
============================

Features describe the state at time T. Targets describe the *outcome* the
model must predict - which by definition lies in the future (T+1 ... T+H).
This is the ONLY module that uses forward shifts (shift(-N)), and these
columns are never fed back into the model as inputs.

Regression targets
------------------
- target_next_close:    Close at T+1 (absolute level - non-stationary; useful
                        for some baselines but rarely the best target).
- target_next_return:   log return from T to T+1. Stationary, the standard
                        regression target.
- target_return_5d:     log return from T to T+5. Multi-horizon momentum.

Classification targets
----------------------
- target_direction:     1 if next-day return > threshold else 0. Matches the
                        `binary_direction` target in data_config.yaml.
- target_signal:        Buy / Hold / Sell (2/1/0) bucketed by next-day return
                        magnitude vs a threshold. Captures "big up", "flat",
                        "big down" - more informative than binary when the
                        class balance allows it.

Label-design trade-offs
-----------------------
- Threshold choice controls class balance + signal strength. threshold=0 =>
  ~50/50 binary but the "up" class includes meaningless +0.01% days.
- Longer horizons are smoother (less label noise) but delay the feedback
  signal and overlap (5-day targets on consecutive days share 4 days).
- Buy/Hold/Sell thresholds should be set from the return *distribution*
  (e.g. +/- one sigma), not picked arbitrarily - EDA (Session 9) informs this.

Leakage status: targets are forward-looking BY DESIGN. The contract is that
downstream code excludes any `target_*` column from the feature matrix X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_target_features(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    direction_threshold: float = 0.0,
    signal_threshold: float = 0.01,
) -> pd.DataFrame:
    """Append regression and classification target columns.

    Args:
        df:                   OHLCV frame with DatetimeIndex.
        price_col:            Canonical adjusted price.
        direction_threshold:  Min next-day return to label "up" (binary).
        signal_threshold:     Abs return magnitude separating Hold from Buy/Sell.

    Returns:
        New DataFrame with `target_*` columns added. Input not mutated.
        The last row(s) will have NaN targets (no future data) - dropped later.
    """
    out = df.copy()
    price = out[price_col]

    # --- Regression targets ------------------------------------------------
    # shift(-1) = tomorrow's value aligned to today's row. Forward-looking.
    out["target_next_close"] = price.shift(-1)
    out["target_next_return"] = np.log(price.shift(-1) / price)
    out["target_return_5d"] = np.log(price.shift(-5) / price)

    # --- Binary classification (matches data_config: binary_direction) -----
    out["target_direction"] = (
        out["target_next_return"] > direction_threshold
    ).astype("Int64")  # nullable int so NaNs stay NaN, not 0

    # --- 3-class Buy/Hold/Sell --------------------------------------------
    # 2 = Buy (big up), 1 = Hold (flat), 0 = Sell (big down). Threshold applied
    # to the absolute next-day return.
    r = out["target_next_return"]
    signal = np.where(r > signal_threshold, 2,
              np.where(r < -signal_threshold, 0, 1))
    out["target_signal"] = pd.array(signal, dtype="Int64")

    return out
