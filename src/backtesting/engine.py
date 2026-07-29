"""
StockM v1.0 - Phase 8, Lesson 2 (skeleton)
Backtesting Engine (orchestrator)
=================================

Single Responsibility: DRIVE the simulation - wire strategy -> risk ->
portfolio together and walk the prediction + price timeline one bar at a time.
The engine owns NO domain logic: it is a thin conductor. All intelligence lives
in the collaborating modules; the engine just enforces ORDER and the "never
use the future" timing contract.

Why a thin orchestrator
-----------------------
A "fat engine" that knows about stops, sizing, and costs becomes the one module
everything must change to extend - the opposite of SRP. A thin engine means:
    - new strategy   -> subclass Strategy        (engine untouched)
    - new risk rule  -> edit RiskManager         (engine untouched)
    - new cost model -> edit Portfolio           (engine untouched)
The engine's only job is the loop and the result assembly.

Event-driven (not vectorized): the config sets ``type: event_driven``. We loop
bar-by-bar so stops, trailing exits, and multi-asset position logic are exact.
For 765 days x 50 tickers (~38k iterations) this is trivially fast in Python.

Timing contract (the engine's invariant)
----------------------------------------
At bar t, the engine:
    1. marks open positions to t's price                  (mark-to-market)
    2. asks the RiskManager for forced exits at t's price (stops checked)
    3. asks the Strategy for a signal using data <= t       (no look-ahead)
    4. gates the signal through the RiskManager            (risk limits)
    5. executes the (possibly vetoed) signal via Portfolio at t's price
Decisions use <= t; execution is at t's price (Adj Close). Step ordering
matters: exits before entries so freed capital is available the same bar.

Dependency: imports strategy, risk_management, portfolio, metrics, trade.
Implemented incrementally: the loop body fills in as Lessons 5 / 7 land; the
interface + BacktestResult are fixed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.risk_management import RiskManager
from backtesting.strategy import Strategy
from backtesting.trade import Trade

logger = logging.getLogger("stockm.backtesting.engine")


@dataclass
class BacktestResult:
    """Everything one backtest produces, bundled for metrics + reporting.

    Carries provenance (config + metadata) so a saved result is reproducible:
    given the same predictions + config + model version, the numbers must
    match (Lesson 13).

    Attributes:
        strategy_name:     Name of the strategy that produced this run.
        symbols:           Ticker(s) traded in this backtest.
        equity_curve:      Portfolio value indexed by date (the core artifact).
        trades:            Ordered list of executed Trades.
        metrics:           Computed performance metrics (from metrics.py).
        config:            The backtest config used (for reproducibility).
        benchmark_equity:  Optional buy-and-hold equity curve for comparison.
        metadata:          Free-form provenance: model version, seeds, etc.
    """

    strategy_name: str
    symbols: list[str]
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: dict[str, float]
    config: dict
    benchmark_equity: pd.Series | None = None
    metadata: dict = field(default_factory=dict)


class BacktestEngine:
    """Thin event-driven orchestrator of the simulation loop."""

    def __init__(
        self,
        strategy: Strategy,
        portfolio: Portfolio,
        risk_manager: RiskManager,
        config: dict,
    ) -> None:
        self.strategy = strategy
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.config = config
        self.logger = logging.getLogger("stockm.backtesting.engine")

    def run(
        self, predictions: pd.DataFrame, prices: pd.DataFrame
    ) -> "BacktestResult":
        """Walk the timeline bar-by-bar and return the full BacktestResult.

        Args:
            predictions: Date-indexed table of model predictions per symbol
                         (columns: predicted_return, [confidence]).
            prices:       Date-indexed canonical price (Adj Close) per symbol.
        """
        raise NotImplementedError("Lesson 5 (wired as portfolio lands)")
