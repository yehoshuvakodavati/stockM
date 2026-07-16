"""
StockM v1.0  -  Phase 2: Data Collection
Entry point for the historical-data batch ingestion system.

Role of this file (the "conductor"):
    It decides WHAT to download - the ticker universe (a Python list) -
    and will later call the collector (src/collectors/historical_collector.py)
    to perform each download.

Today (Lesson 1) this file only defines the ticker list and prints a few
facts about it. In Lesson 2 we add a loop; in Lesson 3 we connect that loop
to the collector.

Run from the project root:
    python src/main.py
"""

# ---------------------------------------------------------------------------
# The ticker universe
# ---------------------------------------------------------------------------
# A Python list: an ordered, mutable (changeable) collection of values.
# We use it to hold the set of stock symbols we want to download.
#
# Every item is a string (text), and every item is an ".NS" ticker, meaning
# the stock trades on the NSE (National Stock Exchange of India). Keeping a
# single type of value in one list is a professional habit - it makes the
# list predictable and easy to loop over.
tickers = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]


def main() -> None:
    """Print basic facts about the ticker list (Lesson 1 demonstration)."""
    print(f"Ticker universe : {tickers}")
    print(f"Number of stocks: {len(tickers)}")
    print(f"First stock     : {tickers[0]}")   # index 0  = first item
    print(f"Last stock      : {tickers[-1]}")  # index -1 = last item


if __name__ == "__main__":
    main()
