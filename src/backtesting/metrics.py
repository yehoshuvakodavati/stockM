"""
StockM v1.0 - Phase 8, Lesson 2 (skeleton)
Performance Metrics
====================

Single Responsibility: turn an equity curve + trade log into performance
statistics. PURE FUNCTIONS, no state, no I/O. This makes them trivially
unit-testable (feed a known equity curve, assert the Sharpe) and reusable
across backtests, paper trading, and live monitoring.

Why metrics are pure functions
------------------------------
A metric that reads files, holds state, or mutates inputs is impossible to
trust: you can never be sure which run produced which number. Pure functions
over (equity_curve, trades) -> dict make every number reproducible and let
Lesson 10's strategy comparison call the SAME function for every strategy
(apples-to-apples).

Annualisation: NSE has ~252 trading days/year (overridable per call). The
daily risk-free rate defaults to 0 for v1; Sharpe / Sortino annualise from the
daily frequency. See Lesson 8.

Dependency: imports backtesting.trade (Trade) + numpy / pandas.
Implemented in: Lesson 8.
"""

from __future__ import annotations

import logging

import pandas as pd

from backtesting.trade import Trade

logger = logging.getLogger("stockm.backtesting.metrics")

# NSE convention: ~252 trading sessions per year. Overridable per call so a
# future crypto (365) or weekly horizon can reuse the same functions.
TRADING_DAYS_PER_YEAR = 252

# The stable metric contract. Lesson 10's comparison and Lesson 12's report
# both rely on these exact keys existing in the dict compute_metrics returns.
METRIC_NAMES = [
    "total_return",
    "cagr",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "profit_factor",
    "win_rate",
    "avg_profit",
    "avg_loss",
    "expectancy",
    "turnover",
    "n_trades",
]


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[Trade],
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute the full metric suite from an equity curve + trade log.

    Returns a flat dict with every key in :data:`METRIC_NAMES`. Each component
    is also exposed as a standalone pure function (implemented in Lesson 8) so
    individual metrics can be asserted in tests.
    """
    raise NotImplementedError("Lesson 8")
