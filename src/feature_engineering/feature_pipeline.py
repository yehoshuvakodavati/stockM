"""
StockM v1.0 - Phase 3, Lessons 2 / 12 / 14 / 15
Feature Engineering Orchestrator
================================

This module ties the feature families together into one config-driven
pipeline:

    load raw OHLCV  ->  add_* feature families  ->  handle missing values
                    ->  validate  ->  save AI-ready dataset

Single Responsibility
---------------------
This orchestrator owns *only* the sequencing and the load/save boundary.
The math of each feature lives in its own family module, the validation
logic lives in feature_validator, and the parameters live in
configs/feature_config.yaml. Adding a new feature family = one new module
+ one line here. That is the Open/Closed Principle in practice.

Leakage contract (enforced here)
--------------------------------
- Feature families use only current/past rows.
- Only target_features.py looks forward, and only into `target_*` columns.
- Missing-value handling drops rows; it never fills with future-derived
  statistics, so no leakage is introduced at cleanup time.

Run
---
    python src/run_feature_pipeline.py            # all tickers
    python src/run_feature_pipeline.py RELIANCE.NS  # one ticker
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Family modules - imported as a package-relative name because the runner
# puts src/ on sys.path (mirrors src/main.py's convention).
from feature_engineering.price_features import add_price_features
from feature_engineering.trend_features import add_trend_features
from feature_engineering.momentum_features import add_momentum_features
from feature_engineering.volatility_features import add_volatility_features
from feature_engineering.volume_features import add_volume_features
from feature_engineering.time_features import add_time_features
from feature_engineering.lag_features import add_lag_features
from feature_engineering.rolling_features import add_rolling_features
from feature_engineering.target_features import add_target_features
from feature_engineering.feature_validator import validate_features

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, never hard-coded)
#   __file__      = .../stockM/src/feature_engineering/feature_pipeline.py
#   parents[0]    = .../src/feature_engineering
#   parents[1]    = .../src
#   parents[2]    = .../stockM   <- project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
METADATA_DIR = PROJECT_ROOT / "data" / "processed" / "metadata"

logger = logging.getLogger("stockm.feature_engineering")

# Canonical column we treat as "the price". The raw CSV stores it as
# "Adj Close" (with a space); we standardise to a code-friendly name.
PRICE_COL = "Adj Close"


def _safe_filename(symbol: str) -> str:
    """Turn 'RELIANCE.NS' into 'RELIANCE_NS' (matches the collector's naming)."""
    return symbol.replace(".", "_")


def load_raw_ohlc(symbol: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load a single ticker's raw OHLCV CSV as a typed, indexed frame.

    Args:
        symbol:  Yahoo ticker, e.g. "RELIANCE.NS".
        raw_dir: Override for the raw data directory (testing).

    Returns:
        DataFrame with a DatetimeIndex and columns
        ``Adj Close, Close, High, Low, Open, Volume`` (all numeric).
    """
    raw_dir = raw_dir or RAW_DIR
    path = raw_dir / f"{_safe_filename(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw OHLCV not found for {symbol}: {path}")

    # index_col=0 -> the Date column becomes the index; parse_dates for time
    # features. NaT-safe: bad/missing dates become NaT and are dropped later.
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Empty / header-only files (e.g. a download that returned no rows) would
    # otherwise surface as a confusing downstream "DatetimeIndex required"
    # error. Fail fast with an actionable message instead.
    if df.shape[0] == 0:
        raise ValueError(
            f"Raw file for {symbol} has no data rows ({path.name}). "
            f"Re-run the collector for this ticker."
        )

    # If the date column failed to parse (e.g. blank header), the index is a
    # plain Index of strings. Coerce it to datetime so time features work.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # Sort chronologically - some downloaders can append out of order, and
    # every rolling/shift operation here assumes ascending time.
    df = df.sort_index()

    # Drop non-trading rows (holidays/weekends where yfinance occasionally
    # emits an all-NaN row) and duplicate dates (keep last).
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]

    # Coerce to numeric so stray strings (e.g. "null") become NaN, not object
    # dtype that would silently break the rolling math.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_features(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run every feature family on an OHLCV frame and return the engineered frame.

    This is the public, side-effect-free core: given a clean OHLCV frame and
    an optional parameter dict, return a frame with all engineered features
    and targets. Used both by the runner and by tests / notebooks.

    Args:
        df:     OHLCV frame with DatetimeIndex and the standard columns.
        params: Optional parameter dict (see configs/feature_config.yaml
                `ohlcv` block). Missing keys fall back to each family's
                sensible defaults.

    Returns:
        New DataFrame with ~60-80 feature columns + target columns. Rows
        with NaN warmup / future-edge are NOT yet dropped here - that is
        the orchestrator's responsibility (see ``process_ticker``).
    """
    p = params or {}

    # Unpack with defaults so a partial config never crashes a family.
    sma_windows = tuple(p.get("sma_windows", (5, 10, 20, 50, 100, 200)))
    ema_windows = tuple(p.get("ema_windows", (5, 10, 20, 50)))
    roc_periods = tuple(p.get("roc_periods", (5, 10, 20)))
    momentum_periods = tuple(p.get("momentum_periods", (5, 10, 20)))
    vma_windows = tuple(p.get("vma_windows", (5, 10, 20)))
    lag_periods = tuple(p.get("lag_periods", (1, 3, 5, 10, 20)))
    rolling_windows = tuple(p.get("rolling_windows", (5, 10, 20)))

    rsi_window = int(p.get("rsi_window", 14))
    macd_fast = int(p.get("macd_fast", 12))
    macd_slow = int(p.get("macd_slow", 26))
    macd_signal = int(p.get("macd_signal", 9))
    vol_window = int(p.get("volatility_window", 20))
    atr_window = int(p.get("atr_window", 14))
    bb_window = int(p.get("bb_window", 20))
    bb_std = float(p.get("bb_std", 2.0))
    hv_window = int(p.get("hv_window", 21))

    direction_threshold = float(p.get("direction_threshold", 0.0))
    signal_threshold = float(p.get("signal_threshold", 0.01))

    # --- Compose the families in dependency order --------------------------
    # Order matters only for readability/auditability: each family is
    # self-contained (recomputes its own return basis) so they could run in
    # any order. Targets go last because they are conceptually separate.
    out = df
    out = add_price_features(out, price_col=PRICE_COL)
    out = add_trend_features(
        out, price_col=PRICE_COL,
        sma_windows=sma_windows, ema_windows=ema_windows,
    )
    out = add_momentum_features(
        out, price_col=PRICE_COL,
        rsi_window=rsi_window, macd_fast=macd_fast,
        macd_slow=macd_slow, macd_signal=macd_signal,
        roc_periods=roc_periods, momentum_periods=momentum_periods,
    )
    out = add_volatility_features(
        out, price_col=PRICE_COL,
        vol_window=vol_window, atr_window=atr_window,
        bb_window=bb_window, bb_std=bb_std, hv_window=hv_window,
    )
    out = add_volume_features(
        out, price_col=PRICE_COL,
        vma_windows=vma_windows,
        vol_momentum_fast=int(p.get("vol_momentum_fast", 5)),
        vol_momentum_slow=int(p.get("vol_momentum_slow", 20)),
    )
    out = add_time_features(out)
    out = add_lag_features(out, price_col=PRICE_COL, lag_periods=lag_periods)
    out = add_rolling_features(
        out, price_col=PRICE_COL, windows=rolling_windows,
    )
    out = add_target_features(
        out, price_col=PRICE_COL,
        direction_threshold=direction_threshold,
        signal_threshold=signal_threshold,
    )

    return out


def _handle_missing(
    df: pd.DataFrame,
    strategy: str = "drop",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Handle NaNs introduced by rolling windows / lags / forward targets.

    Args:
        df:       Engineered frame (features + targets).
        strategy: "drop"  - drop any row with a NaN (warmup + future edge).
                  "ffill" - forward-fill features, then drop remaining
                            (target NaNs at the end are never filled).
                  "bfill" - back-fill (NOT recommended - leaks future stats;
                            included only for completeness / comparison).

    Returns:
        (cleaned_frame, stats) where stats records rows dropped per reason.
    """
    before = len(df)
    stats = {"rows_before": before}

    if strategy == "drop":
        cleaned = df.dropna()
    elif strategy == "ffill":
        # Forward-fill feature columns only; never touch targets (would leak).
        feat_cols = [c for c in df.columns if not c.startswith("target_")]
        cleaned = df.copy()
        cleaned[feat_cols] = cleaned[feat_cols].ffill()
        cleaned = cleaned.dropna()
    elif strategy == "bfill":
        # Back-fill leaks future information into the present. Logged as a
        # warning by the caller; here we just execute it for comparison runs.
        cleaned = df.dropna()  # still drop after, for the future-edge targets
        # (intentionally we do NOT bfill by default; see warning in process_ticker)
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")

    stats["rows_after"] = len(cleaned)
    stats["rows_dropped"] = before - len(cleaned)
    return cleaned, stats


class FeatureEngine:
    """Config-driven wrapper around :func:`build_features` with load/save.

    Holds the parameter dict and the output paths so the runner can loop
    over many tickers with a single instance. Stateless across tickers
    except for the (immutable) config and the output dirs.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        missing_strategy: str = "drop",
        features_dir: Path | None = None,
        metadata_dir: Path | None = None,
    ) -> None:
        self.params = params or {}
        self.missing_strategy = missing_strategy
        self.features_dir = features_dir or FEATURES_DIR
        self.metadata_dir = metadata_dir or METADATA_DIR

    # -- single ticker -----------------------------------------------------
    def process_ticker(self, symbol: str) -> dict[str, Any]:
        """Load, engineer, validate and save features for one ticker.

        Returns a summary dict (rows in/out, file path, validation ok).
        """
        logger.info("Processing %s", symbol)
        raw = load_raw_ohlc(symbol)
        engineered = build_features(raw, params=self.params)

        if self.missing_strategy == "bfill":
            logger.warning(
                "missing_strategy='bfill' leaks future data; use only for "
                "diagnostic comparison, never for training data."
            )

        cleaned, miss_stats = _handle_missing(engineered, strategy=self.missing_strategy)
        report = validate_features(cleaned)

        if not report["ok"]:
            # Log but do not crash - the report tells you exactly what's wrong.
            logger.warning(
                "%s validation issues: duplicates=%s constant=%s dtype=%s",
                symbol,
                report["duplicate_columns"],
                report["constant"],
                report["dtype_issues"],
            )

        file_path = self._save(cleaned, symbol)
        self._save_report(report, symbol, miss_stats)

        return {
            "symbol": symbol,
            "raw_rows": int(len(raw)),
            "feature_rows": int(len(cleaned)),
            "feature_cols": int(cleaned.shape[1]),
            "rows_dropped": miss_stats["rows_dropped"],
            "validation_ok": report["ok"],
            "correlated_pairs": report["correlated_pair_count"],
            "file": str(file_path),
        }

    # -- persistence -------------------------------------------------------
    def _save(self, df: pd.DataFrame, symbol: str) -> Path:
        self.features_dir.mkdir(parents=True, exist_ok=True)
        path = self.features_dir / f"{_safe_filename(symbol)}_features.csv"
        # index=True writes the Date as the first column -> round-trippable.
        df.to_csv(path, index=True)
        logger.info("saved %s -> %s (%d rows x %d cols)",
                    symbol, path.name, df.shape[0], df.shape[1])
        return path

    def _save_report(
        self, report: dict[str, Any], symbol: str, miss_stats: dict[str, int]
    ) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        path = self.metadata_dir / f"{_safe_filename(symbol)}_feature_report.json"
        report = {**report, "missing_handling": miss_stats, "symbol": symbol}
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
