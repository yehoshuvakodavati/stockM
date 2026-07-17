"""
StockM v1.0 - Phase 2: Data Collection
Layer 1: Historical Market Data

historical_collector.py - the single-stock download engine.

This module contains ONE public responsibility: download the historical
OHLCV data for a given stock symbol and save it as a raw CSV.

    main.py  ->  historical_collector.download_stock(symbol)  ->  data/raw/<SYMBOL>.csv

No exception handling, no validation, no logging yet - those arrive in the
Session 6 lessons. This is the honest baseline we will harden step by step.
"""

from pathlib import Path

import yfinance as yf


# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, never hard-coded)
#   __file__      = .../stockM/src/collectors/historical_collector.py
#   parents[0]    = .../src/collectors
#   parents[1]    = .../src
#   parents[2]    = .../stockM   <- project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Default download window. Hard-coded for now; a later lesson moves these
# into configs/data_config.yaml so the whole project shares one source of truth.
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2026-07-12"


def _flatten_columns(df) -> None:
    """Collapse yfinance's MultiIndex columns ('Close', 'TICKER') -> 'Close'.

    yfinance >= 1.5 returns a 2-level column index even for a single ticker.
    For one symbol the ticker level is redundant noise, so we drop it.
    The leading underscore in the name marks this as an internal helper.
    """
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)


def _safe_filename(symbol: str) -> str:
    """Turn 'RELIANCE.NS' into 'RELIANCE_NS' so the dot isn't a fake extension."""
    return symbol.replace(".", "_")


def download_stock(symbol: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> Path:
    """
    Download daily OHLCV history for one symbol and save it as a raw CSV.

    Args:
        symbol: Yahoo Finance ticker, e.g. "RELIANCE.NS".
        start:  Start date (inclusive), "YYYY-MM-DD".
        end:    End date (exclusive in yfinance), "YYYY-MM-DD".

    Returns:
        The Path of the CSV file that was written.

    Side effects:
        Creates data/raw/ if missing and writes <SYMBOL>.csv into it.
    """
    # Ensure the raw directory exists (idempotent: no error if it already does)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,   # keep Adj Close (adjusted for splits + dividends)
        progress=False,      # silence yfinance's progress bar in batch runs
    )

    # 2. Clean the column structure
    _flatten_columns(df)

    # 3. Persist to data/raw/<SYMBOL>.csv  (Date index written as first column)
    file_path = RAW_DIR / f"{_safe_filename(symbol)}.csv"
    df.to_csv(file_path, index=True)

    return file_path
