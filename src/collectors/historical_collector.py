"""
StockM v1.0 - Phase 2: Data Collection
Layer 1: Historical Market Data

historical_collector.py - the single-stock download engine.

This module contains ONE public responsibility: download the historical
OHLCV data for a given stock symbol and save it as a raw CSV.

    main.py  ->  historical_collector.download_stock(symbol)  ->  data/raw/<SYMBOL>.csv

Resilience features added after the JSPL/TATAMOTORS Yahoo outages:
  - Symbol-alias fallback: if the primary symbol returns no data (Yahoo
    intermittently 404s individual .NS symbols), configured aliases are
    tried in order (e.g. the BSE numeric code 532286.BO for JSPL).
  - Empty-result detection: a 0-row download raises a clear error instead
    of silently writing a header-only file (which previously broke the
    feature pipeline downstream with a confusing "DatetimeIndex" message).
"""

import logging
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

logger = logging.getLogger("stockm.collector")

# Symbol-alias fallback map.
# Yahoo Finance intermittently 404s certain .NS symbols (returns "possibly
# delisted; no timezone found" with zero rows). When that happens we retry
# with an alternate symbol that resolves to the SAME company - typically the
# BSE numeric code (e.g. JSPL = Jindal Steel & Power = 532286.BO). Aliases
# are only used when the primary returns empty, never preemptively.
SYMBOL_ALIASES: dict[str, list[str]] = {
    "JSPL.NS": ["532286.BO"],          # Jindal Steel & Power (BSE code)
    "TATAMOTORS.NS": ["500570.BO"],    # Tata Motors (BSE code); Yahoo still 404-ing as of 2026-07-21
}


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


def _fetch(symbol: str, start: str, end: str):
    """Single yfinance download call. Returns the (possibly empty) frame."""
    return yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,   # keep Adj Close (adjusted for splits + dividends)
        progress=False,      # silence yfinance's progress bar in batch runs
        threads=False,       # serial fetches - gentler on Yahoo's rate limiter
    )


def download_stock(symbol: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> Path:
    """
    Download daily OHLCV history for one symbol and save it as a raw CSV.

    Tries the primary symbol first; if Yahoo returns zero rows, falls back to
    any configured aliases (see SYMBOL_ALIASES). The file is always saved
    under the *primary* symbol's name so downstream pipelines address it
    consistently regardless of which alias actually served the data.

    Args:
        symbol: Yahoo Finance ticker, e.g. "RELIANCE.NS".
        start:  Start date (inclusive), "YYYY-MM-DD".
        end:    End date (exclusive in yfinance), "YYYY-MM-DD".

    Returns:
        The Path of the CSV file that was written.

    Raises:
        ValueError: if the symbol and all its aliases return zero rows.

    Side effects:
        Creates data/raw/ if missing and writes <SYMBOL>.csv into it.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Candidates in priority order: the primary symbol, then its aliases.
    candidates = [symbol] + SYMBOL_ALIASES.get(symbol, [])

    df = None
    used = symbol
    for cand in candidates:
        fetched = _fetch(cand, start, end)
        _flatten_columns(fetched)
        if fetched.shape[0] > 0:
            df = fetched
            used = cand
            break
        logger.warning("%s: %s returned 0 rows", symbol, cand)

    if df is None:
        # Refuse to write a header-only file - that previously caused a
        # confusing "DatetimeIndex required" error deep in the feature
        # pipeline. Fail fast with an actionable message instead.
        raise ValueError(
            f"No data for {symbol} (tried {candidates}). Yahoo may be "
            f"temporarily 404-ing this symbol; retry later or add an alias "
            f"in SYMBOL_ALIASES."
        )

    if used != symbol:
        logger.info("%s: served by alias %s", symbol, used)

    file_path = RAW_DIR / f"{_safe_filename(symbol)}.csv"
    df.to_csv(file_path, index=True)
    return file_path
