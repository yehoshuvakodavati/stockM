"""
StockM v1.0 - Phase 5, Lesson 12
Prediction Pipeline
===================

Reusable inference functions that load a saved model and produce predictions
on unseen data. The four public functions mirror the roadmap:

    load_model(symbol)                 -> (model, metadata)
    predict(model, X)                  -> y_hat
    predict_single_stock(symbol, ...)  -> {date, predicted_return, signal, ...}
    predict_batch(symbols, ...)        -> {symbol: prediction, ...}

Signal generation
-----------------
A predicted return is turned into a trading signal by thresholding:
    y_hat > +threshold  -> BUY
    y_hat < -threshold  -> SELL
    otherwise           -> HOLD
The threshold is configurable; default 0.0 gives pure direction. In practice
you'd tune it against transaction costs and the return distribution (Session 9
EDA informs the threshold).

Data note
---------
These functions predict on the PREPARED (already-selected, already-scaled)
feature rows - i.e. the held-out test split is the "unseen data" demonstrated
here. For LIVE inference on today's raw OHLCV, the production path is:
    raw OHLCV -> feature_engineering -> select the saved 40 features ->
    scale with the saved scaler -> predict. The scaler and feature list are
    referenced in each model's metadata (scaler_ref / feature_names) so a
    future live-inference module can reproduce the exact transform.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from models.data_loader import load_split, split_xy
from models.model_storage import SAVED_MODELS_DIR, load_model as _load_from_storage

logger = logging.getLogger("stockm.models.prediction")

# Default signal threshold on predicted return.
DEFAULT_SIGNAL_THRESHOLD = 0.0


def _signal_from_return(ret: float, threshold: float) -> str:
    if ret > threshold:
        return "BUY"
    if ret < -threshold:
        return "SELL"
    return "HOLD"


def load_model(symbol: str, model_name: str | None = None):
    """Load a saved model + metadata for a symbol (best model if name is None)."""
    return _load_from_storage(symbol, model_name)


def predict(model, X: pd.DataFrame) -> pd.Series:
    """Generate predictions for a feature matrix. Returns y_hat aligned to X."""
    return pd.Series(model.predict(X), index=X.index, name="predicted_return")


def predict_single_stock(
    symbol: str,
    date: str | None = None,
    split: str = "test",
    model_name: str | None = None,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    target_col: str = "target_next_return",
    prepared_dir: Path | None = None,
) -> dict[str, Any]:
    """Predict next-day return + signal for one stock on a given date.

    Args:
        symbol:    Ticker, e.g. "RELIANCE.NS".
        date:      ISO date to predict for; None = latest available in the split.
        split:     Which prepared split to predict on (default "test" = unseen).
        model_name: Specific model, or None to use the deployed best model.
        threshold: Signal threshold on predicted return.
        target_col: Target the model was trained on (for showing realised y).

    Returns:
        Dict with symbol, date, predicted_return, signal, and (if available)
        the realised target for hindsight comparison.
    """
    model, metadata = load_model(symbol, model_name)
    df = load_split(symbol, split, prepared_dir)

    # Restrict to the model's trained feature set (guard against drift).
    feature_names = metadata.get("feature_names", [])
    if feature_names:
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            raise KeyError(f"Features missing from {split} split: {missing}")
        X = df[feature_names]
    else:
        X, _ = split_xy(df, target_col)

    # Pick the row: a specific date, else the latest available.
    if date is not None:
        ts = pd.Timestamp(date)
        if ts not in X.index:
            # Find the nearest available date <= requested (never future).
            prior = X.index[X.index <= ts]
            if len(prior) == 0:
                raise KeyError(f"No data on/before {date} for {symbol} in {split}")
            ts = prior[-1]
        X_row = X.loc[[ts]]
    else:
        ts = X.index[-1]
        X_row = X.iloc[[-1]]

    y_hat = float(model.predict(X_row)[0])
    signal = _signal_from_return(y_hat, threshold)

    # Hindsight: the realised target (only known AFTER the day - never used
    # as an input). Useful for evaluating the signal.
    realised = None
    if target_col in df.columns and ts in df.index:
        realised = float(df.loc[ts, target_col])

    return {
        "symbol": symbol,
        "date": str(ts.date()),
        "model": metadata.get("model_name", model_name),
        "predicted_return": round(y_hat, 6),
        "signal": signal,
        "realised_return": (round(realised, 6) if realised is not None else None),
        "correct_direction": (
            (y_hat > 0) == (realised > 0) if realised is not None else None
        ),
    }


def predict_batch(
    symbols: list[str],
    date: str | None = None,
    split: str = "test",
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    target_col: str = "target_next_return",
    prepared_dir: Path | None = None,
) -> dict[str, dict[str, Any] | str]:
    """Predict for many symbols. Returns {symbol: prediction_or_error}."""
    out: dict[str, Any] = {}
    for sym in symbols:
        try:
            out[sym] = predict_single_stock(
                sym, date=date, split=split, threshold=threshold,
                target_col=target_col, prepared_dir=prepared_dir,
            )
        except Exception as e:  # noqa: BLE001 - batch must continue
            logger.warning("predict failed for %s: %s", sym, e)
            out[sym] = {"symbol": sym, "error": str(e)}
    return out
