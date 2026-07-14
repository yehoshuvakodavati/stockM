"""
StockM v1 - Layer 1: Historical Market Data
Phase 2: Historical Data Collection

What this script does (one symbol at a time):
    1. Downloads daily OHLCV history for a single stock symbol from Yahoo Finance.
    2. Inspects the resulting Pandas DataFrame so we understand its shape,
       columns, index, and data types.
    3. Flattens yfinance's MultiIndex columns into simple, flat columns.
    4. Saves the RAW, unmodified data to data/raw/ as a CSV file.

Run it from the project root:
    python src/stockM/download_data.py
"""

from pathlib import Path

import yfinance as yf


# ---------------------------------------------------------------------------
# 1. Configuration
#    These values are hard-coded for now so we can learn the mechanics.
#    Later, in a future lesson, we will read them from configs/data_config.yaml
#    and configs/paths.yaml so that no path or symbol is ever hard-coded.
# ---------------------------------------------------------------------------
SYMBOL = "RELIANCE.NS"     # Yahoo Finance ticker for Reliance Industries (NSE)
START_DATE = "2005-05-05"  # inclusive
END_DATE = "2026-07-12"    # exclusive in yfinance (data up to, but not, this date)


# ---------------------------------------------------------------------------
# 2. Resolve the raw-data directory relative to the project root.
#
#    __file__        = .../stockM/src/stockM/download_data.py  (this file)
#    .resolve()      = the same path, but absolute and with no ".." left in it
#    .parents[0]     = .../stockM/src/stockM
#    .parents[1]     = .../stockM/src
#    .parents[2]     = .../stockM                       <- the project root
#
#    We never hard-code "C:\\stockM" because that would break on another
#    machine. Resolving from __file__ keeps the script portable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw if it is missing


def download_symbol(symbol: str, start: str, end: str):
    """
    Download daily OHLCV history for one symbol and return a DataFrame.

    auto_adjust=False keeps the classic 6 columns (Open, High, Low, Close,
    Adj Close, Volume). Adj Close is the price adjusted for splits and
    dividends - it is the price we will use later to compute returns.
    """
    return yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )


def flatten_columns(df) -> None:
    """
    yfinance >= 1.5 returns a MultiIndex column DataFrame even for one symbol:
        columns look like ('Close', 'RELIANCE.NS'), ('High', 'RELIANCE.NS'), ...

    For a single symbol that extra ticker level is noise. We keep only the
    top level ('Open', 'High', ...). This modifies the DataFrame in place.

    df.columns.nlevels is the number of levels in the column index.
    1 = a normal flat index, 2 = a MultiIndex (the yfinance case).
    """
    if df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)


def inspect_data(df, symbol: str) -> None:
    """Print a structured summary so we understand what we downloaded."""
    print("=" * 70)
    print(f"  Symbol : {symbol}")
    print(f"  Range  : {START_DATE}  ->  {END_DATE}")
    print("=" * 70)

    # df.shape is a tuple: (number_of_rows, number_of_columns)
    print(f"Shape (rows, cols) : {df.shape}")

    # The column labels (the 'variables' / 'features' in ML vocabulary)
    print(f"Columns            : {list(df.columns)}")

    # The index is the row labels. For stock data it is the trading dates.
    print(f"Index type         : {type(df.index).__name__}")
    print(f"Index name         : {df.index.name}")
    print(f"Index range        : {df.index.min().date()}  ->  {df.index.max().date()}")

    # dtypes = data types. float64 = decimal numbers, int64 = whole numbers.
    print("\nData types:")
    print(df.dtypes)

    # Peek at the first and last rows to sanity-check the values.
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nLast 3 rows:")
    print(df.tail(3))

    # describe() gives count, mean, std, min, max and quartiles per column.
    print("\nSummary statistics:")
    print(df.describe())


def save_raw_csv(df, symbol: str, start: str, end: str) -> Path:
    """
    Save the raw DataFrame to data/raw/ and return the file path.

    File-naming convention:  <SYMBOL>_<interval>_<start>_<end>.csv
        RELIANCE.NS  ->  RELIANCE_NS_1d_2005-05-05_2026-07-12.csv

    The '.' in the ticker is replaced with '_' so it is not mistaken for the
    file extension. The interval (1d = daily) and the date range are baked
    into the name so two different downloads never silently overwrite each
    other.
    """
    safe_symbol = symbol.replace(".", "_")
    file_name = f"{safe_symbol}_1d_{start}_{end}.csv"
    file_path = RAW_DIR / file_name

    # index=True writes the Date index as the first column of the CSV.
    df.to_csv(file_path, index=True)
    return file_path


def main() -> None:
    # 1. Download
    df = download_symbol(SYMBOL, START_DATE, END_DATE)

    # 2. Flatten the MultiIndex columns into simple flat columns
    flatten_columns(df)

    # 3. Inspect
    inspect_data(df, SYMBOL)

    # 4. Persist the RAW data to data/raw/
    file_path = save_raw_csv(df, SYMBOL, START_DATE, END_DATE)
    print("\n" + "=" * 70)
    print(f"Saved raw data to: {file_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
