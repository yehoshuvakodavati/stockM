"""
StockM v1.0 - Phase 3: Feature Engineering
==========================================

Transforms clean OHLCV market data into an AI-ready feature dataset.

Public API:
    build_features   - generate the full feature set for one ticker's OHLCV frame
    FeatureEngine    - class wrapping the config-driven pipeline (used by the runner)

Package layout (one module per feature family - Single Responsibility Principle):
    price_features.py       - returns, log returns, ranges, typical/weighted price
    trend_features.py       - SMA / EMA families
    momentum_features.py    - RSI, MACD, ROC, momentum
    volatility_features.py  - rolling std, ATR, Bollinger Bands, historical vol
    volume_features.py      - volume change, VMA, volume ratio, OBV
    time_features.py        - calendar / ordinal time components
    lag_features.py         - lagged close / open / volume / returns
    rolling_features.py     - rolling mean / max / min / median / variance
    target_features.py      - next-day close/return, 5-day return, B/H/S labels
    feature_validator.py    - NaN / inf / duplicate / constant / correlation checks
    feature_pipeline.py     - orchestrator: load -> generate -> validate -> save

Design contract (the iron rule of this package):
    For every feature on the row dated T, it MUST be computable using only
    data from day T and earlier. No feature peeks into the future - except
    the explicitly-named target columns, which are the only forward-looking
    fields and are never used as model inputs.
"""

from feature_engineering.feature_pipeline import FeatureEngine, build_features

__all__ = ["FeatureEngine", "build_features"]
