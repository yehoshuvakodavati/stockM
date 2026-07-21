"""
StockM v1.0 - Phase 2: Data Collection
Entry point for the historical-data batch ingestion system.

Flow:
    read config/tickers.csv  ->  for each symbol  ->  download_stock(symbol)
                                                    ->  data/raw/<SYMBOL>.csv

The batch is resilient to individual ticker failures: if Yahoo 404s a symbol
(and all its aliases), that ticker is logged and skipped - the run continues
for the rest of the universe instead of aborting on the first failure.

Run from the project root:
    python src/main.py
"""

import csv
import logging
import sys
from pathlib import Path

# Make the 'collectors' package importable when running `python src/main.py`.
# Python puts the script's own directory (src/) on sys.path automatically,
# so `from collectors...` resolves once src/collectors/__init__.py exists.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collectors.historical_collector import download_stock


# Project root = this file's parent directory (src/main.py -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TICKERS_CSV = PROJECT_ROOT / "config" / "tickers.csv"

# Configure logging so the collector's warnings (e.g. "JSPL.NS returned 0
# rows", "served by alias 532286.BO") are actually surfaced to the console
# and a log file. Without this, those logger calls are silently dropped.
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("stockm.ingestion")


def load_tickers(csv_path: Path) -> list[str]:
    """
    Read ticker symbols from a one-column CSV with a 'Symbol' header.

    Returns a list of strings, e.g. ["RELIANCE.NS", "TCS.NS", ...].
    Blank lines and surrounding whitespace are ignored.
    """
    symbols: list[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["Symbol"].strip()
            if symbol:                       # skip empty lines
                symbols.append(symbol)
    return symbols


def main() -> int:
    """
    Download every ticker in the universe, skipping (not aborting on) failures.

    Returns:
        0 if every ticker succeeded, 1 if any ticker failed. The exit code
        lets downstream schedulers / CI detect partial failures.
    """
    logger.info("Reading ticker universe from: %s", TICKERS_CSV)
    tickers = load_tickers(TICKERS_CSV)
    logger.info("Found %d symbols. Starting batch download...", len(tickers))

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for ticker in tickers:
        try:
            file_path = download_stock(ticker)
            logger.info("  saved %-16s -> %s", ticker, file_path.name)
            succeeded.append(ticker)
        except ValueError as e:
            # Expected failure: Yahoo returned no data for the symbol or any
            # alias. Skip it and keep going - one bad ticker must not abort
            # the whole batch.
            failed.append((ticker, str(e)))
            logger.warning("  SKIP  %-16s %s", ticker, e)
        except Exception as e:  # noqa: BLE001 - batch run must stay alive
            # Unexpected failure (network, disk, ...). Log the traceback so it
            # is diagnosable, then continue with the next ticker.
            failed.append((ticker, repr(e)))
            logger.exception("  FAIL  %-16s unexpected error", ticker)

    logger.info(
        "\nDone. %d/%d succeeded | %d skipped/failed.",
        len(succeeded), len(tickers), len(failed),
    )
    for sym, err in failed:
        logger.info("  FAILED %s: %s", sym, err)

    return 0 if not failed else 1


if __name__ == "__main__":
    # Exit with main()'s return code so partial failures are visible to the
    # shell / scheduler (e.g. `python src/main.py && next-step`).
    raise SystemExit(main())
