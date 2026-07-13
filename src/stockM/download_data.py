import yfinance as yf
from yfinance import data

data = yf.download("RELIANCE.NS", "2005-05-05", "2026-07-12")

print(data)