"""
StockM v1.0 - Phase 2: Data Collection
Entry point for the historical-data batch ingestion system.

Flow:
    read config/tickers.csv  ->  for each symbol  ->  download_stock(symbol)
                                                    ->  data/raw/<SYMBOL>.csv

Run from the project root:
    python src/main.py
"""

import csv
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


def main() -> None:
    print(f"Reading ticker universe from: {TICKERS_CSV}")
    tickers = load_tickers(TICKERS_CSV)
    print(f"Found {len(tickers)} symbols. Starting batch download...\n")

    for ticker in tickers:
        file_path = download_stock(ticker)
        print(f"  saved {ticker:16s} -> {file_path.name}")

    print(f"\nDone. Downloaded {len(tickers)} stocks.")


if __name__ == "__main__":
    main()
