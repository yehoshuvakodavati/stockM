"""
StockM v1.0 - Phase 7, Lesson 2
Sequential Data Preparation
===========================

Converts the prepared tabular datasets (one feature row per trading day) into
the temporal *windows* that recurrent / convolutional / attention models eat.

    data/prepared/<SYM>/{train,validation,test}.csv   (rows = days)
        -> load_dataset()           (leakage-safe X/y split, Phase 5)
        -> build_sequences()        (THIS MODULE: rows -> windows)
        -> X_train_seq : (n_windows, lookback, n_features)
           y_train_seq : (n_windows,)

Why sequences are required for recurrent models
-----------------------------------------------
A tree model (RandomForest / XGBoost) sees ONE row at a time: the lags it uses
are hand-engineered columns (return_lag_1, return_lag_3, ...). It has no notion
of *order* beyond what we froze into those columns. An LSTM/GRU/Transformer, by
contrast, ingests an ordered block of consecutive days and learns the temporal
shape itself - e.g. "a 5-day momentum thrust followed by a volume spike tends to
precede a reversal." That shape lives in the *sequence*, not in any single row,
so we must hand the model windows, not rows.

Leakage discipline (the part that silently kills finance models)
----------------------------------------------------------------
1. Per-split windowing. Train windows are built from train rows ONLY, val from
   val rows, test from test rows. A window never spans two splits, so no
   split's input can peek at another split's data. (The tabular baselines were
   trained the same per-split way; this keeps the Lesson-10 comparison fair.)
2. Causal pairing. Window i = feature rows [i .. i+lookback-1]; its target is
   y[i+lookback-1+(horizon-1)], i.e. a value realised AFTER the window's last
   feature row. The model never sees a target that belongs to the past.
3. Target firewall inherited. ``load_dataset`` already strips every ``target_*``
   column from X (asserted), so no target - not even a sibling like
   target_return_5d - can hide inside a window's features.
4. Gap respected. The prepared splits already carry a 1-day no-leakage gap
   (gap_days=1); per-split windowing preserves it.

Trade-off we accept (and report, never hide)
--------------------------------------------
The first ``lookback-1`` rows of each split cannot start a full window and are
dropped as warmup. For RELIANCE (lookback=30) that is 29 rows per split - ~0.8%
of train, ~3.8% of val/test. The loss is small and HONEST; we surface the count
in the returned dict and the logs.

A future enhancement ("Option B") would let val/test windows reach back into
train for realistic recent context (a deployed model at val_start genuinely has
train history). That is NOT leakage (past context is allowed), but it couples
splits and complicates the fair comparison, so we defer it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Reuse the Phase-5 leakage-safe loader: identical data/target/split as baselines.
from models.data_loader import PREPARED_DIR, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONFIG = PROJECT_ROOT / "configs" / "feature_config.yaml"

logger = logging.getLogger("stockm.deep_learning.sequence_builder")

# DL frameworks default to float32; float64 doubles memory and triggers dtype
# warnings. We cast features to float32 at the window boundary (the natural
# hand-off point to Keras/PyTorch).
_DEFAULT_DTYPE = np.float32


@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for sequence generation.

    Attributes
    ----------
    lookback : int
        Number of consecutive trading days per input window (= sequence length
        fed to the model). Read from ``feature_config.yaml`` -> ``windowing.lookback_window``.
    horizon : int
        Forecast horizon in days. 1 = predict the next-day outcome recorded at
        the window's last row. >1 shifts the target further ahead.
    stride : int
        Step between consecutive window starts. 1 = every possible window
        (maximum samples, heavy overlap - the right default for small datasets).
        >1 subsamples windows (less overlap, fewer samples, faster).
    target_col : str
        Column used as y. Target-agnostic: ``target_next_return`` (regression,
        matches the baselines) or ``target_direction`` (binary UP/DOWN). The
        builder does not care - the model (Lesson 4+) decides the loss/head.
    """

    lookback: int = 30
    horizon: int = 1
    stride: int = 1
    target_col: str = "target_next_return"

    @classmethod
    def from_feature_config(cls, config_path: Path | None = None) -> "SequenceConfig":
        """Build a SequenceConfig from ``feature_config.yaml`` (config-driven).

        Reads the ``windowing`` block (lookback_window, horizon). Missing keys
        fall back to the dataclass defaults, so a trimmed config still runs.
        """
        path = config_path or FEATURE_CONFIG
        cfg: dict[str, Any] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        w = cfg.get("windowing", {})
        return cls(
            lookback=int(w.get("lookback_window", cls.lookback)),
            horizon=int(w.get("horizon", cls.horizon)),
            stride=int(w.get("stride", cls.stride)),
            # target_col is DL-specific; feature_config has no DL block yet, so
            # we keep the baseline-matching default. Override via build_sequences().
            target_col=cls.target_col,
        )

    def validate(self) -> None:
        """Fail fast on nonsensical params (cheaper than a confusing crash later)."""
        if self.lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self.lookback}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if not self.target_col.startswith("target_"):
            raise ValueError(
                f"target_col must be a target_* column, got {self.target_col!r}"
            )


# ---------------------------------------------------------------------------
# Low-level windowing: arrays in, windows out. Pure function, easy to test.
# ---------------------------------------------------------------------------
def _build_windows(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    lookback: int,
    horizon: int,
    stride: int,
    dtype: type = _DEFAULT_DTYPE,
) -> tuple[np.ndarray, np.ndarray, pd.Index | None]:
    """Slice a 2-D feature matrix into overlapping windows + aligned targets.

    Args:
        X:        Feature matrix, shape (n_rows, n_features). DataFrame preferred
                  (its index is returned as per-window forecast dates).
        y:        Target vector, shape (n_rows,).
        lookback: Window length (rows).
        horizon:  Forecast steps beyond the window's last row.
        stride:   Step between window starts.
        dtype:    Numeric dtype for X windows (float32 by default).

    Returns:
        (X_seq, y_seq, dates) where
            X_seq  : (n_windows, lookback, n_features)
            y_seq  : (n_windows,)
            dates  : index of the row each forecast pertains to (None if X has
                     no index). Length == n_windows.

    Raises:
        ValueError: if there are too few rows to form even one window.
    """
    # Capture the index BEFORE converting to a raw array (it carries the dates).
    index: pd.Index | None = getattr(X, "index", None)

    X_arr = np.ascontiguousarray(np.asarray(X, dtype=dtype))
    y_arr = np.asarray(y, dtype=np.float64)  # keep target precision; model casts later

    n_rows, n_features = X_arr.shape
    if y_arr.shape[0] != n_rows:
        raise ValueError(
            f"X and y row mismatch: X has {n_rows}, y has {y_arr.shape[0]}"
        )

    # Number of windows: window i spans [i, i+lookback-1]; its target lives at
    # i+lookback-1+(horizon-1), which must stay within [0, n_rows-1].
    n_windows = n_rows - lookback - horizon + 2
    if n_windows <= 0:
        raise ValueError(
            f"Too few rows ({n_rows}) for lookback={lookback}, horizon={horizon}; "
            f"need >= {lookback + horizon - 1}."
        )

    # sliding_window_view slides along axis 0 and appends the window axis LAST:
    #   (n_rows-lookback+1, n_features, lookback)
    # We keep the first n_windows (later starts have no valid target), then
    # reorder to the Keras/PyTorch convention (samples, timesteps, features).
    all_windows = np.lib.stride_tricks.sliding_window_view(
        X_arr, window_shape=lookback, axis=0
    )
    X_seq = np.ascontiguousarray(all_windows[:n_windows].transpose(0, 2, 1))

    # Target index for window i = its last feature row + (horizon-1).
    target_idx = np.arange(n_windows) + (lookback - 1) + (horizon - 1)
    y_seq = y_arr[target_idx]

    # Stride subsamples windows (default stride=1 keeps every window).
    if stride > 1:
        X_seq = X_seq[::stride]
        y_seq = y_seq[::stride]
        target_idx = target_idx[::stride]

    dates = index[target_idx] if index is not None else None

    # Invariant checks (cheap; catch reshape bugs early rather than at training).
    assert X_seq.shape == (len(target_idx), lookback, n_features), "X_seq shape wrong"
    assert y_seq.shape[0] == X_seq.shape[0], "y/X window count mismatch"
    return X_seq, y_seq, dates


# ---------------------------------------------------------------------------
# High-level entry point: ticker -> sequential train/val/test.
# ---------------------------------------------------------------------------
def build_sequences(
    symbol: str,
    config: SequenceConfig | None = None,
    prepared_dir: Path | None = None,
    target_col: str | None = None,
    dtype: type = _DEFAULT_DTYPE,
) -> dict[str, Any]:
    """Build sequential (windowed) train/val/test arrays for one ticker.

    Consumes ``models.data_loader.load_dataset`` so the DL track trains on the
    SAME leakage-safe X/y/split as the classical baselines (apples-to-apples).

    Args:
        symbol:       Ticker, e.g. "RELIANCE.NS".
        config:       SequenceConfig (lookback/horizon/stride/target_col). If
                      None, read from ``feature_config.yaml``.
        prepared_dir: Override the default prepared directory.
        target_col:   Override config.target_col (e.g. "target_direction").
                      Useful for the regression-vs-classification fork without
                      touching the config.
        dtype:        Feature dtype (float32 default).

    Returns:
        Dict with X/y sequence arrays for each split, plus shapes, window
        counts, per-split warmup-row loss, feature names, and the inherited
        date ranges / NaN-drop count from load_dataset.

    Raises:
        FileNotFoundError: if no prepared dataset exists for ``symbol``.
        ValueError:        on inconsistent shapes or bad config.
    """
    cfg = config or SequenceConfig.from_feature_config()
    cfg.validate()
    tgt = target_col or cfg.target_col

    logger.info(
        "Building sequences | symbol=%s | target=%s | lookback=%d horizon=%d stride=%d",
        symbol, tgt, cfg.lookback, cfg.horizon, cfg.stride,
    )

    # 1) Leakage-safe tabular load (Phase 5). Identical rows/target as baselines.
    base = prepared_dir or PREPARED_DIR
    data = load_dataset(symbol, target_col=tgt, prepared_dir=base)
    feats = data["feature_names"]

    # 2) Window each split independently (per-split = leak-free; see module doc).
    splits_in = [
        ("train", data["X_train"], data["y_train"]),
        ("validation", data["X_val"], data["y_val"]),
        ("test", data["X_test"], data["y_test"]),
    ]
    out: dict[str, Any] = {
        "symbol": symbol,
        "target_col": tgt,
        "feature_names": feats,
        "n_features": data["n_features"],
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "dropped_na_train": data["dropped_na_train"],
        "date_ranges": data["date_ranges"],
        "shapes": {},
        "window_counts": {},
        "rows_dropped_to_windowing": {},  # warmup rows lost per split
    }
    for name, X, y in splits_in:
        n_before = len(X)
        X_seq, y_seq, dates = _build_windows(
            X, y, cfg.lookback, cfg.horizon, cfg.stride, dtype
        )
        out[f"X_{name}_seq"] = X_seq
        out[f"y_{name}_seq"] = y_seq
        out[f"dates_{name}"] = dates
        out["shapes"][name] = X_seq.shape
        out["window_counts"][name] = X_seq.shape[0]
        out["rows_dropped_to_windowing"][name] = n_before - X_seq.shape[0]

    # 3) Cross-split invariants: same feature count, target aligned to windows.
    n_feat = data["n_features"]
    for name in ("train", "validation", "test"):
        Xs = out[f"X_{name}_seq"]
        assert Xs.shape[1:] == (cfg.lookback, n_feat), (
            f"{name} window shape {Xs.shape[1:]} != ({cfg.lookback}, {n_feat})"
        )
        assert out[f"y_{name}_seq"].shape[0] == Xs.shape[0], (
            f"{name}: y windows {out[f'y_{name}_seq'].shape[0]} != X windows {Xs.shape[0]}"
        )

    # 4) Confirm no target column slipped into the feature set (defense in depth;
    #    load_dataset already asserts this, but a regression here is catastrophic).
    assert not any(c.startswith("target_") for c in feats), "target leaked into features!"

    logger.info(
        "%s sequences | train=%s val=%s test=%s | dropped_to_windowing=%s",
        symbol,
        out["shapes"]["train"], out["shapes"]["validation"], out["shapes"]["test"],
        out["rows_dropped_to_windowing"],
    )
    return out


# ---------------------------------------------------------------------------
# Smoke test: build sequences for one ticker and print the leakage-relevant
# facts. Run:  python src/deep_learning/sequence_builder.py RELIANCE.NS
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    seq = build_sequences(sym)
    print("\n=== Sequence summary ===")
    print(f"symbol            : {seq['symbol']}")
    print(f"target_col        : {seq['target_col']}")
    print(f"lookback/horizon  : {seq['lookback']} / {seq['horizon']}")
    print(f"n_features        : {seq['n_features']}")
    for s in ("train", "validation", "test"):
        print(
            f"{s:<10} X={str(seq['shapes'][s]):<22} "
            f"y={seq[f'y_{s}_seq'].shape}  "
            f"dropped={seq['rows_dropped_to_windowing'][s]}"
        )
    print(f"date_ranges       : {seq['date_ranges']}")
    # Quick sanity: a window's last feature row must precede its target's date.
    dtr = seq["dates_train"]
    print(f"first train window forecast date : {dtr[0]}")
    print(f"last  train window forecast date : {dtr[-1]}")
