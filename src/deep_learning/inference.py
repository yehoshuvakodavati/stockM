"""
StockM v1.0 - Phase 7, Lesson 12
Deep-Learning Production Prediction Pipeline
============================================

The DL inference layer: load a saved DL model and produce BUY/HOLD/SELL signals
on unseen data. Mirrors the four-function contract of models.prediction (the ML
path) so the two are interchangeable from a caller's perspective.

Functions (the roadmap's contract)
----------------------------------
    load_deep_learning_model(symbol)        -> (model, metadata)
    predict(model, X_seq, ...)              -> y_hat              [batch of windows]
    predict_single_stock(symbol, ...)       -> {date, predicted_return, signal, ...}
    predict_batch(symbols, ...)             -> {symbol: prediction, ...}

The DL-specific wrinkle
-----------------------
ML models predict on a ROW (40 features, one day). DL models predict on a
WINDOW (lookback x 40, `lookback` days). So predict_single_stock_dl must build
the single window ending at the requested date, not just grab one row. We reuse
sequence_builder.build_sequences (DRY) and index the window whose end-date
matches (or is the latest <= the requested date) - the same "never use the
future" rule as the ML predictor.

Scope (honest, mirrors the ML layer)
------------------------------------
Demonstrated on the PREPARED test split (the held-out "unseen" data). The
production LIVE path is: today's raw OHLCV -> feature_engineering -> select the
saved 40 features -> scale with the saved scaler -> build the window -> predict.
The metadata's feature_names + scaler_ref make that reproducible; wiring the
live feature transform end-to-end is a post-Phase-7 deployment task.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from deep_learning.dl_storage import load_dl_model

logger = logging.getLogger("stockm.deep_learning.inference")

DEFAULT_SIGNAL_THRESHOLD = 0.0


def _signal_from_return(ret: float, threshold: float) -> str:
    if ret > threshold:
        return "BUY"
    if ret < -threshold:
        return "SELL"
    return "HOLD"


def load_deep_learning_model(
    symbol: str, model_name: str | None = None, device: str | None = None
) -> tuple["torch.nn.Module", dict[str, Any]]:
    """Load a saved DL model + metadata, ready for inference.

    Args:
        symbol:     Ticker.
        model_name:  Specific model, or None for the deployed (best_dl_model.json).
        device:      "cpu" / "cuda". None = current device.

    Returns:
        (model in eval mode, metadata). Metadata includes arch_type, target_col,
        feature_names, metrics, dataset, scaler_ref.
    """
    return load_dl_model(symbol, model_name=model_name, device=device)


def predict(
    model: "torch.nn.Module",
    X_seq: np.ndarray | torch.Tensor,
    device: str = "cpu",
    task: str = "regression",
) -> np.ndarray:
    """Generate predictions for a batch of windows.

    Args:
        model: A loaded DL model (eval mode).
        X_seq: (n_windows, lookback, n_features) array/tensor.
        device: "cpu" / "cuda".
        task:   "regression" (raw values) or "classification" (sigmoid -> UP prob).

    Returns:
        (n_windows,) numpy array of predictions.
    """
    model.eval()
    if not isinstance(X_seq, torch.Tensor):
        X_seq = torch.as_tensor(X_seq, dtype=torch.float32)
    X_seq = X_seq.to(device)
    with torch.no_grad():
        out = model(X_seq).squeeze(-1).cpu()
        if task == "classification":
            out = torch.sigmoid(out)
    return out.numpy()


def predict_single_stock(
    symbol: str,
    date: str | None = None,
    split: str = "test",
    model_name: str | None = None,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
    device: str | None = None,
) -> dict[str, Any]:
    """Predict next-day return + signal for one stock using its deployed DL model.

    Builds the window ending at `date` (or the latest available), runs the model,
    and returns the same dict shape as models.prediction.predict_single_stock so
    the two paths are interchangeable.

    Args:
        symbol:    Ticker, e.g. "RELIANCE.NS".
        date:      ISO date to predict for; None = latest window in the split.
        split:     Which prepared split to predict on (default "test" = unseen).
        model_name: Specific DL model, or None for the deployed best.
        threshold: Signal threshold on predicted return.
        device:    "cpu" / "cuda".

    Returns:
        Dict with symbol, date, model, arch_type, predicted_return, signal,
        realised_return (if available), correct_direction.
    """
    from deep_learning.framework import get_device
    from deep_learning.sequence_builder import build_sequences
    from models.data_loader import load_split

    dev = device or get_device("auto")

    # 1) Load the model + metadata (tells us the target_col + arch).
    model, meta = load_deep_learning_model(symbol, model_name=model_name, device=dev)
    arch_type = meta.get("arch_type", "unknown")
    target_col = meta.get("target_col", "target_next_return")
    task = meta.get("arch_kwargs", {}).get("task", "regression")

    # 2) Build the sequences for this split (DRY: reuse the verified builder).
    seq = build_sequences(symbol, target_col=target_col)
    split_key = "validation" if split == "validation" else ("train" if split == "train" else "test")
    X_seq = seq[f"X_{split_key}_seq"]
    dates = seq[f"dates_{split_key}"]

    # 3) Pick the window: a specific date, else the latest. Never future.
    if date is not None:
        ts = pd.Timestamp(date)
        prior = dates[dates <= ts]
        if len(prior) == 0:
            raise KeyError(f"No window ending on/before {date} for {symbol} in {split}")
        target_date = prior[-1]
    else:
        target_date = dates[-1]
    idx = dates.get_loc(target_date)

    # 4) Run the model on the single window -> prediction.
    window = X_seq[idx:idx + 1]                            # (1, lookback, n_features)
    y_hat = float(predict(model, window, device=dev, task=task)[0])
    signal = _signal_from_return(y_hat, threshold)

    # 5) Hindsight: the realised target at the window's end date (known only
    #    AFTER the day - never an input). For honesty, same as the ML path.
    realised = None
    df = load_split(symbol, split)
    if target_col in df.columns and target_date in df.index:
        realised = float(df.loc[target_date, target_col])

    return {
        "symbol": symbol,
        "date": str(pd.Timestamp(target_date).date()),
        "model": meta.get("model_name", model_name),
        "arch_type": arch_type,
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
    device: str | None = None,
) -> dict[str, dict[str, Any] | str]:
    """Predict for many symbols using each one's deployed DL model.

    Returns {symbol: prediction_or_error}. A failure for one symbol never stops
    the batch (same resilience as models.prediction.predict_batch).
    """
    out: dict[str, Any] = {}
    for sym in symbols:
        try:
            out[sym] = predict_single_stock(
                sym, date=date, split=split, threshold=threshold, device=device
            )
        except FileNotFoundError as e:
            out[sym] = {"symbol": sym, "error": f"no DL model: {e}"}
            logger.warning("DL predict failed for %s: %s", sym, e)
        except Exception as e:  # noqa: BLE001 - batch must continue
            out[sym] = {"symbol": sym, "error": str(e)}
            logger.warning("DL predict failed for %s: %s", sym, e)
    return out


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    print("=== DL predict_single_stock ===")
    r = predict_single_stock(sym)
    for k, v in r.items():
        print(f"  {k}: {v}")
    print("\n=== DL predict_batch (RELIANCE + a ticker with no DL model) ===")
    batch = predict_batch([sym, "TCS.NS"])
    for s, p in batch.items():
        print(f"  {s}: {p}")
