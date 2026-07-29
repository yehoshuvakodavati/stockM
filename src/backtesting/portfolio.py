"""
StockM v1.0 - Phase 8, Lesson 2 (skeleton)
Portfolio Simulation
=====================

Single Responsibility: track capital, hold positions, and execute Signals into
Trades - the accounting layer. It owns CASH, OPEN POSITIONS, the EQUITY CURVE,
and the realized / unrealized P&L split. It contains NO decision logic (that's
the strategy) and NO risk limits (that's the risk manager): it faithfully
records what it is told to do.

Why accounting is isolated from decisions
-----------------------------------------
A portfolio that also decides when to trade is impossible to audit: a loss
could come from a bad signal, bad sizing, or a bookkeeping bug. Isolating
accounting means the equity curve is a pure, auditable consequence of the
signals + costs fed in. Costs (Lesson 6) plug in here as a CostModel so the
portfolio stays agnostic to the fee structure until one is supplied.

Dependency: imports backtesting.trade (Signal, Trade, TradeAction). Position
state lives here (it is live portfolio state, not a vocabulary record).
Implemented in: Lesson 5 (portfolio), Lesson 6 (transaction-cost CostModel).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from backtesting.trade import Signal, Trade, TradeAction

logger = logging.getLogger("stockm.backtesting.portfolio")


@dataclass
class Position:
    """A currently-held position (mutable; owned and mutated by Portfolio).

    Unlike Trade / Signal (immutable records of intent / history), a Position
    is live state: its quantity and average entry update on every partial
    fill, and its unrealized P&L moves with the market. This mutability is the
    portfolio's job, so Position lives here rather than in trade.py.

    Attributes:
        symbol:               Ticker held.
        quantity:             Shares currently held (>= 0; long-only in v1).
        avg_entry_price:      Volume-weighted average buy price.
        highest_since_entry:  Peak price since entry (trailing-stop ref,
                              used by RiskManager in Lesson 7).
        realized_pnl:         P&L booked on the parts of this position already
                              closed (accumulates across partial sells).
    """

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    highest_since_entry: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        """True if the position currently holds any shares."""
        return self.quantity > 0


class Portfolio:
    """Cash + positions + equity curve; executes Signals into Trades."""

    def __init__(self, initial_capital: float, config: dict) -> None:
        raise NotImplementedError("Lesson 5")

    def execute(self, signal: Signal, price: float) -> Trade | None:
        """Apply one signal at ``price``; return the booked Trade or None."""
        raise NotImplementedError("Lesson 5")

    def mark_to_market(self, prices: dict[str, float], dt: date) -> float:
        """Value the book at current prices; append today's equity to the curve."""
        raise NotImplementedError("Lesson 5")

    @property
    def cash(self) -> float:
        raise NotImplementedError("Lesson 5")

    @property
    def equity_curve(self) -> pd.Series:
        raise NotImplementedError("Lesson 5")

    @property
    def trades(self) -> list[Trade]:
        raise NotImplementedError("Lesson 5")
