"""
StockM v1.0 - Phase 8, Lesson 2 (skeleton)
Risk Management
================

Single Responsibility: act as a GATE between strategy signals and execution.
Given a proposed Signal and the current portfolio state, the RiskManager may:
    - pass it through unchanged,
    - reshape it (e.g. cap the position size to max_position),
    - veto it (return None), or
    - generate its OWN exit signals (stop-loss / take-profit / trailing stop)
      independent of the model.

Why risk is a separate gate, not embedded in the portfolio
----------------------------------------------------------
Mixing "what's the risk limit?" with "execute the fill" violates SRP and makes
risk rules untestable in isolation. Professionals want to swap risk regimes
(conservative vs aggressive) without touching portfolio accounting. Keeping
it as a pure function of (signal, state) -> signal | None makes every risk
rule unit-testable with synthetic states - no price data required. See
Lesson 7.

Dependency: imports backtesting.trade; references Portfolio only under
TYPE_CHECKING (to avoid a circular import: Portfolio owns execution, Risk
owns limits - neither should import the other at runtime).
Implemented in: Lesson 7.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

from backtesting.trade import Signal

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from backtesting.portfolio import Portfolio

logger = logging.getLogger("stockm.backtesting.risk")


class RiskManager:
    """Enforce risk constraints on every signal and manage exit triggers.

    Constraints (all configurable, Lesson 7):
        stop_loss_pct, take_profit_pct, trailing_stop_pct,
        max_risk_per_trade_pct, max_position_pct, max_daily_loss_pct.

    Two duties:
        1. apply(signal, portfolio)            -> gate an ENTRY signal.
        2. check_exits(positions, prices, dt)  -> forced EXIT signals.
    """

    def __init__(self, config: dict) -> None:
        raise NotImplementedError("Lesson 7")

    def apply(self, signal: Signal, portfolio: "Portfolio") -> Signal | None:
        """Gate an entry signal: return it (possibly reshaped) or None to veto."""
        raise NotImplementedError("Lesson 7")

    def check_exits(
        self, positions: dict, prices: dict[str, float], dt: date
    ) -> list[Signal]:
        """Generate forced exit signals for stops / triggers hit this bar."""
        raise NotImplementedError("Lesson 7")
