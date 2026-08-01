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

Cost-model contract
-------------------
The portfolio never hard-codes a fee structure. It accepts any object with::

    cost_model.estimate(notional: float, action: TradeAction, price: float) -> float

returning the total transaction cost (commission + slippage + taxes) for a
trade of the given notional. Lesson 6 ships a real ``CostModel``; today a
``NullCostModel`` (zero cost) is the default so the portfolio is fully usable
and testable before costs exist. Swapping the model is one constructor arg.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd

from backtesting.trade import Signal, Trade, TradeAction

logger = logging.getLogger("stockm.backtesting.portfolio")


class CostModel(Protocol):
    """Structural interface for transaction-cost models (Lesson 6 fills this in).

    A Protocol (structural typing) rather than an ABC: any object with a
    matching ``estimate`` method satisfies it - no inheritance required. This
    lets Lesson 6 add ``FixedCostModel`` / ``PercentageCostModel`` without
    touching this file, and lets tests pass a bare lambda.
    """

    def estimate(self, notional: float, action: TradeAction, price: float) -> float: ...


class NullCostModel:
    """Zero transaction cost. The default until Lesson 6 supplies a real model.

    Useful for isolating strategy quality from cost drag (run a backtest with
    NullCostModel, then with PercentageCostModel, and the delta is pure cost).
    """

    def estimate(self, notional: float, action: TradeAction, price: float) -> float:  # noqa: D401
        return 0.0


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
    """Cash + positions + equity curve; executes Signals into Trades.

    The accounting layer. It owns three things and nothing else:
      1. CASH balance        - spendable capital (rises on sells, falls on buys).
      2. OPEN POSITIONS      - one mutable ``Position`` per symbol held.
      3. EQUITY CURVE        - a date-indexed Series of total portfolio value.

    It contains NO decision logic (that is the strategy's job) and NO risk
    limits (that is the RiskManager's job in Lesson 7): given a Signal and a
    price, it faithfully records the trade and its cost. Isolating accounting
    means the equity curve is a pure, auditable consequence of the signals +
    costs fed in - a loss can always be traced to a signal, sizing, or a cost,
    never to a bookkeeping bug.

    Position sizing (v1)
    --------------------
    Long-only, single position per symbol. A BUY that adds to an existing
    position uses the *signal's strength* to scale how much fresh capital to
    deploy (Lesson 4 reserved ``strength`` for exactly this). Fractional shares
    are allowed (equities are continuous) so the portfolio never stalls on
    "can't afford a whole share". A SELL with no open position is a no-op
    (logged) rather than an error - the strategy may not know the book state.

    Args:
        initial_capital: Starting cash (e.g. 1_000_000).
        cost_model:      Transaction-cost model (Lesson 6). None -> NullCostModel.
        config:          Optional dict of sizing knobs; recognised keys:
                          ``max_position_pct`` (cap per buy as a fraction of
                          equity, default 1.0 = all-in), ``allow_fractional``
                          (default True).
    """

    def __init__(
        self,
        initial_capital: float,
        cost_model: CostModel | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        self._initial_capital = float(initial_capital)
        self._cash = float(initial_capital)
        self._cost_model: CostModel = cost_model if cost_model is not None else NullCostModel()
        cfg = config or {}
        self._max_position_pct = float(cfg.get("max_position_pct", 1.0))
        self._allow_fractional = bool(cfg.get("allow_fractional", True))

        # Live state.
        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        # Equity curve: (date, total_equity). Appended once per mark_to_market.
        self._equity_records: list[tuple[date, float]] = []

        # Cached realised P&L across the whole portfolio (sum of closed trades).
        self._realized_pnl = 0.0

    # ------------------------------------------------------------------ state

    @property
    def initial_capital(self) -> float:
        """The cash the portfolio started with (for total-return metrics)."""
        return self._initial_capital

    @property
    def cash(self) -> float:
        """Current spendable cash."""
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """Currently-open positions (symbol -> Position). Read-only view."""
        return self._positions

    @property
    def trades(self) -> list[Trade]:
        """Every booked Trade, in execution order (the trade log)."""
        return list(self._trades)

    @property
    def realized_pnl(self) -> float:
        """Cumulative realised P&L from all closed positions."""
        return self._realized_pnl

    @property
    def equity_curve(self) -> pd.Series:
        """Date-indexed total portfolio value (cash + market value of positions)."""
        if not self._equity_records:
            return pd.Series(dtype=float, name="equity")
        idx, vals = zip(*self._equity_records)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")

    def open_positions_value(self, prices: dict[str, float]) -> float:
        """Mark-to-market value of all open positions at the given prices."""
        total = 0.0
        for sym, pos in self._positions.items():
            if pos.is_open and sym in prices:
                total += pos.quantity * prices[sym]
        return total

    def total_equity(self, prices: dict[str, float]) -> float:
        """Cash + market value of open positions (the account value)."""
        return self._cash + self.open_positions_value(prices)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Unrealised P&L across open positions at the given prices.

        unrealised = market_value - cost_basis, where cost_basis is the average
        entry price paid for the shares still held.
        """
        total = 0.0
        for sym, pos in self._positions.items():
            if pos.is_open and sym in prices:
                total += (prices[sym] - pos.avg_entry_price) * pos.quantity
        return total

    # ------------------------------------------------------------- execution

    def execute(self, signal: Signal, price: float) -> Trade | None:
        """Apply one Signal at ``price``; return the booked Trade or None.

        This is the single mutator. It decides quantity (position sizing),
        charges the cost, updates cash + the Position, and appends a Trade to
        the log. Returns None for HOLD, no-op SELLs, and zero-quantity fills
        (e.g. not enough cash) so the caller can skip them.

        Args:
            signal: The intended action (BUY/SELL/HOLD) for one symbol.
            price:  Execution price (Adj Close by convention).

        Returns:
            The booked Trade, or None if nothing was transacted.
        """
        if signal.action == TradeAction.HOLD:
            return None
        if price is None or price <= 0 or pd.isna(price):
            logger.warning("skip %s %s: non-positive/NaN price", signal.symbol, signal.action)
            return None

        if signal.action == TradeAction.BUY:
            return self._buy(signal, price)
        if signal.action == TradeAction.SELL:
            return self._sell(signal, price)
        return None  # unreachable; TradeAction has only BUY/SELL/HOLD

    def _buy(self, signal: Signal, price: float) -> Trade | None:
        """Open or add to a long position, scaled by signal strength."""
        sym = signal.symbol
        # Sizing: deploy a fraction of CURRENT equity capped by max_position_pct.
        # equity = cash + held value; using equity (not just cash) lets the
        # strategy add to winners without first selling. Strength in [0,1].
        equity = self._cash + self.open_positions_value({sym: price})
        budget = equity * self._max_position_pct * max(0.0, min(float(signal.strength), 1.0))
        budget = min(budget, self._cash)  # never spend cash we don't have
        if budget <= 0:
            logger.debug("BUY %s skipped: no budget (cash=%.2f)", sym, self._cash)
            return None

        # Reserve for the transaction cost: the total cash spent is
        # ``notional + cost``, and both must fit in ``budget``. Cost is monotonic
        # in notional, so a few proportional-scaling iterations converge for ANY
        # cost model (percentage, fixed, tiered). This guarantees cash >= 0.
        qty = budget / price
        cost = 0.0
        for _ in range(5):
            notional = qty * price
            cost = self._cost_model.estimate(notional, TradeAction.BUY, price)
            spend = notional + cost
            if spend <= budget + 1e-9:
                break
            if spend <= 0:  # degenerate cost model; nothing to buy
                qty = 0.0
                break
            qty *= budget / spend  # shrink so spend ~= budget
        if not self._allow_fractional:
            qty = float(int(qty))  # whole shares only (rounds down -> still fits)
            notional = qty * price
            cost = self._cost_model.estimate(notional, TradeAction.BUY, price)
        if qty <= 0:
            logger.debug("BUY %s skipped: cost exceeds budget", sym)
            return None
        notional = qty * price

        # Settle cash and update the position's average entry (VWAP).
        self._cash -= notional + cost
        pos = self._positions.get(sym)
        if pos is None:
            pos = Position(symbol=sym)
            self._positions[sym] = pos
        new_qty = pos.quantity + qty
        if new_qty > 0:
            pos.avg_entry_price = (
                (pos.avg_entry_price * pos.quantity) + (price * qty)
            ) / new_qty
        pos.quantity = new_qty
        pos.highest_since_entry = max(pos.highest_since_entry, price)

        trade = Trade(
            date=signal.date, symbol=sym, action=TradeAction.BUY,
            quantity=qty, price=price, value=notional, cost=cost, signal=signal,
        )
        self._trades.append(trade)
        logger.debug("BUY  %s qty=%.4f @%.2f cost=%.2f", sym, qty, price, cost)
        return trade

    def _sell(self, signal: Signal, price: float) -> Trade | None:
        """Close (part of) a long position. No short-selling in v1."""
        sym = signal.symbol
        pos = self._positions.get(sym)
        if pos is None or not pos.is_open:
            logger.debug("SELL %s skipped: no open position", sym)
            return None

        # Sizing: strength scales how much of the position to exit (1.0 = all).
        qty = pos.quantity * max(0.0, min(float(signal.strength), 1.0))
        if not self._allow_fractional:
            qty = float(int(qty))
        if qty <= 0:
            return None

        notional = qty * price
        cost = self._cost_model.estimate(notional, TradeAction.SELL, price)
        proceeds = notional - cost
        # Realised P&L on the closed chunk = proceeds minus its cost basis.
        realized = proceeds - (pos.avg_entry_price * qty)

        self._cash += proceeds
        self._realized_pnl += realized
        pos.realized_pnl += realized
        pos.quantity -= qty
        if not pos.is_open:
            # Position fully closed: reset entry tracking so a future re-entry
            # starts clean (avg_entry_price would otherwise leak the old basis).
            pos.avg_entry_price = 0.0
            pos.highest_since_entry = 0.0

        trade = Trade(
            date=signal.date, symbol=sym, action=TradeAction.SELL,
            quantity=qty, price=price, value=notional, cost=cost, signal=signal,
        )
        self._trades.append(trade)
        logger.debug("SELL %s qty=%.4f @%.2n pct=%.4f cost=%.2f realized=%.2f",
                     sym, qty, price, realized, cost, realized)
        return trade

    # ------------------------------------------------------------ valuation

    def mark_to_market(self, prices: dict[str, float], dt: date) -> float:
        """Value the book at current prices; append today's equity to the curve.

        Called once per bar by the engine AFTER any trades for that bar are
        executed. Returns total equity and records (date, equity) so the equity
        curve is complete - metrics (Lesson 8) consume this Series directly.

        Args:
            prices: {symbol: current_price} for every held symbol (and any
                    others; extras are ignored).
            dt:     The bar's date.

        Returns:
            Total portfolio equity at this bar (cash + market value).
        """
        # Track the trailing-stop high for every open position (Lesson 7 hook).
        for sym, pos in self._positions.items():
            if pos.is_open and sym in prices:
                p = prices[sym]
                if p > pos.highest_since_entry:
                    pos.highest_since_entry = p
        equity = self.total_equity(prices)
        self._equity_records.append((dt, equity))
        return equity
