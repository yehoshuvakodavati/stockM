"""
StockM v1.0 - Phase 8: Backtesting & Trading Strategy
=====================================================

Replays prediction models over historical data under realistic costs and risk
constraints, then measures profitability and risk. The bridge from "a model
that predicts" to "a strategy that trades."

Package layout (one responsibility per module - SOLID):
    trade.py            - shared vocabulary: Signal, Trade, TradeAction (data)
    strategy.py         - predictions -> Signals; Strategy ABC + SignalGenerator
                          (open for extension, closed for modification)
    risk_management.py  - gate: enforce risk limits + generate exit signals
    portfolio.py        - cash + positions + equity curve; execute Trades
    metrics.py          - equity curve + trades -> performance stats (pure fn)
    visualization.py    - render publication-quality charts
    engine.py           - thin event-driven orchestrator + BacktestResult
    report_generator.py - assemble a saved, structured report

Design contract
---------------
- Event-driven loop: bar-by-bar so stops, trailing exits, multi-asset logic
  are exact (config: type: event_driven).
- Timing invariant: decisions at bar t use information <= t; execution at t's
  Adj Close. The engine enforces this; strategies must not peek.
- Separation of concerns: prediction (Phase 7) -> Signal (strategy) -> Trade
  (portfolio). Each is swappable independently.
- Costs are a model, not an afterthought (Lesson 6); the portfolio is
  cost-agnostic until a CostModel is plugged in.
- Metrics are pure functions so every strategy is scored apples-to-apples.
- Honest by design: given ~50% directional accuracy, we EXPECT most strategies
  to lose to Buy & Hold after costs. The framework surfaces that, not hides it.

Public API
----------
    BacktestEngine, BacktestResult, Strategy, SignalGenerator,
    Portfolio, Position, RiskManager, Signal, Trade, TradeAction.
"""

from __future__ import annotations

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.portfolio import Portfolio, Position
from backtesting.risk_management import RiskManager
from backtesting.strategy import SignalGenerator, Strategy
from backtesting.trade import Signal, Trade, TradeAction

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Strategy",
    "SignalGenerator",
    "Portfolio",
    "Position",
    "RiskManager",
    "Signal",
    "Trade",
    "TradeAction",
]
