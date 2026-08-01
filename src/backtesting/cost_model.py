"""
StockM v1.0 - Phase 8, Lesson 6
Transaction Cost Models
=======================

Single Responsibility: turn a trade's notional value into a dollar cost. The
portfolio (Lesson 5) calls ``cost_model.estimate(notional, action, price)`` and
subtracts the result from cash; it knows nothing about *how* the cost is built.
This module owns every assumption about fees, slippage, and taxes.

Why costs deserve their own module
----------------------------------
"Ignore costs and every strategy looks profitable." Transaction costs are the
single most common reason a backtest that looks great on paper loses money live:
they are certain (you pay them every trade), they compound (over hundreds of
trades), and they are asymmetric (slippage always hurts you). Isolating them in
a CostModel means you can (a) swap the fee structure without touching the
portfolio, (b) run the same strategy with vs. without costs to measure pure
cost drag, and (c) plug in a broker's exact fee schedule for live trading.

Composable design
-----------------
A real cost is the SUM of independent components - brokerage + exchange charge +
slippage + tax. Each component is a small object implementing ``estimate``; a
``CompositeCostModel`` sums them. Adding a new fee (e.g. a new SEBI levy) = add
one component, never edit existing ones (Open-Closed, same pattern as the
Lesson 4 strategy registry).

Slippage modeling note
----------------------
Slippage is modeled here as a *cost* (a dollar amount = slippage_pct * notional),
not as a slipped execution price. This is mathematically equivalent for the
EQUITY CURVE: total equity = cash + (qty * market_price), and both approaches
move cash by the same total, so the equity curve and every return/risk metric
(Lesson 8) are identical. Only the realised/unrealised P&L *split* differs
slightly (entry slippage is labelled "cost" rather than folded into the entry
price). For v1 backtesting this is the right trade-off - it keeps the portfolio
agnostic without a price-adjustment refactor. A future live-execution path would
model slippage as a price for exact tax accounting.

Config (configs/backtest_config.yaml)
-------------------------------------
    costs:
      commission_pct: 0.1       # percent of trade value  (0.1 -> 0.001 fraction)
      slippage_pct: 0.05        # percent of trade value
      fixed_cost_per_trade: 0.0 # flat $ per trade
Percent fields are stored as the number of percent (0.1 = 0.1%) and converted to
fractions internally, so the config reads the way a trader speaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from backtesting.trade import TradeAction

logger = logging.getLogger("stockm.backtesting.cost_model")


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class CostComponent(Protocol):
    """Structural interface for one fee component (commission, slippage, tax...).

    A Protocol, not an ABC: any object with ``estimate`` satisfies it, so a test
    or a live broker adapter can pass a bare callable-like without inheriting.
    """

    def estimate(self, notional: float, action: TradeAction, price: float) -> float: ...


# ---------------------------------------------------------------------------
# Atomic cost components
# ---------------------------------------------------------------------------

@dataclass
class PercentageCost:
    """A cost that is a fixed percentage of the trade notional (e.g. brokerage).

    The canonical commission model: pay ``rate`` of every trade's value. Most
    discount brokers (US and India) charge a percentage of turnover, often
    capped; the cap is optional (``max_cost``) and models "0.1% or $20,
    whichever is lower" style schedules.

    Attributes:
        rate:      Fraction of notional (0.001 = 0.1%). ALWAYS a fraction here;
                   ``from_percent`` converts a "percent number" for you.
        max_cost:  Optional cap per trade (a broker's "up to $X" ceiling).
        name:      Label for the cost breakdown in reports.
    """

    rate: float
    max_cost: float | None = None
    name: str = "commission"

    @classmethod
    def from_percent(cls, percent: float, max_cost: float | None = None, name: str = "commission") -> "PercentageCost":
        """Build from a percent NUMBER (0.1 -> 0.001 fraction). Config-friendly."""
        return cls(rate=percent / 100.0, max_cost=max_cost, name=name)

    def estimate(self, notional: float, action: TradeAction, price: float) -> float:
        if notional <= 0:
            return 0.0
        cost = self.rate * notional
        if self.max_cost is not None:
            cost = min(cost, self.max_cost)
        return cost


@dataclass
class FixedCost:
    """A flat fee per trade, regardless of size (e.g. per-order brokerage).

    Some brokers (notably Indian flat-fee brokers like Zerodha's equity delivery
    at Rs 0 / intraday at Rs 20) charge a fixed rupee amount per order, not a
    percentage. For small trades this dominates; for large trades it's negligible
    - which is exactly why modelling it matters for realistic sizing.

    Attributes:
        amount:   Flat fee per executed trade (currency units).
        buy_only: If True, charge only on BUYs (some fees are one-sided).
    """

    amount: float
    buy_only: bool = False
    name: str = "fixed"

    def estimate(self, notional: float, action: TradeAction, price: float) -> float:
        if notional <= 0:
            return 0.0
        if self.buy_only and action != TradeAction.BUY:
            return 0.0
        return self.amount


@dataclass
class SlippageCost:
    """Price-impact / slippage: the gap between signal price and fill price.

    When you place a market order you don't get the reference (Adj Close) price -
    you pay worse. Slippage models this as a percentage of notional. It is ALWAYS
    a cost to you (buy high, sell low), so it applies symmetrically to both sides.

    Two modes:
      - ``mode="constant"``: a fixed slippage_pct of notional (simple, common).
      - ``mode="linear_impact"``: slippage scales with trade SIZE relative to a
        reference volume (bigger trades move the market more). Requires the
        caller to pass volume via ``price``'s companion - not used in v1, but the
        hook is here for when volume data is wired through.

    Attributes:
        pct:            Fraction of notional (0.0005 = 0.05%).
        linear_coeff:   For linear_impact mode; slippage = pct + coeff*(size/ref).
    """

    pct: float
    mode: str = "constant"
    linear_coeff: float = 0.0
    name: str = "slippage"

    @classmethod
    def from_percent(cls, percent: float, name: str = "slippage") -> "SlippageCost":
        return cls(pct=percent / 100.0, name=name)

    def estimate(self, notional: float, action: TradeAction, price: float) -> float:
        if notional <= 0:
            return 0.0
        return self.pct * notional  # constant mode (linear_impact reserved)


@dataclass
class TaxCost:
    """Optional transaction taxes (e.g. India STT, US SEC fee).

    Regulatory taxes are usually one-sided (paid on the sell, or on the buy) and
    a tiny percentage. They are "optional" in the config sense (set rate=0 to
    disable) but mandatory in reality - omitting STT on an Indian backtest is a
    common source of phantom alpha.

    Attributes:
        rate:        Fraction of notional.
        side:        "buy", "sell", or "both" - which leg is taxed.
        name:        Label (e.g. "STT", "SEC_fee").
    """

    rate: float
    side: str = "sell"
    name: str = "tax"

    def estimate(self, notional: float, action: TradeAction, price: float) -> float:
        if notional <= 0:
            return 0.0
        if self.side == "buy" and action != TradeAction.BUY:
            return 0.0
        if self.side == "sell" and action != TradeAction.SELL:
            return 0.0
        return self.rate * notional


# ---------------------------------------------------------------------------
# Composite (sum of components)
# ---------------------------------------------------------------------------

@dataclass
class CompositeCostModel:
    """Sum of independent cost components. The production cost model.

    Holds a list of components and estimates total cost = sum of each. The
    breakdown (which component contributed what) is retained for the report
    (Lesson 12) via :meth:`breakdown`.

    Attributes:
        components: Ordered list of CostComponent objects.
    """

    components: list[CostComponent] = field(default_factory=list)

    def estimate(self, notional: float, action: TradeAction, price: float) -> float:
        if notional <= 0:
            return 0.0
        return sum(c.estimate(notional, action, price) for c in self.components)

    def breakdown(self, notional: float, action: TradeAction, price: float) -> dict[str, float]:
        """Per-component cost for a trade - for the report's cost analysis section."""
        if notional <= 0:
            return {}
        return {getattr(c, "name", type(c).__name__): c.estimate(notional, action, price) for c in self.components}


# ---------------------------------------------------------------------------
# Factories: config-driven construction + market presets
# ---------------------------------------------------------------------------

def from_config(config: dict[str, Any]) -> CompositeCostModel:
    """Build a CompositeCostModel from the ``costs:`` section of backtest_config.

    Reads commission_pct, slippage_pct, fixed_cost_per_trade (all stored as
    percent NUMBERS - 0.1 means 0.1%). Unknown keys are ignored so the config
    can grow new fee types without breaking older parsers.

    Args:
        config: The ``costs`` dict from backtest_config.yaml.

    Returns:
        A CompositeCostModel with percentage commission + slippage + fixed fee.
    """
    comm_pct = float(config.get("commission_pct", 0.0))
    slip_pct = float(config.get("slippage_pct", 0.0))
    fixed = float(config.get("fixed_cost_per_trade", 0.0))

    comps: list[CostComponent] = []
    if comm_pct > 0:
        comps.append(PercentageCost.from_percent(comm_pct, name="commission"))
    if slip_pct > 0:
        comps.append(SlippageCost.from_percent(slip_pct, name="slippage"))
    if fixed > 0:
        comps.append(FixedCost(amount=fixed, name="fixed"))
    model = CompositeCostModel(components=comps)
    logger.info(
        "cost model from config: commission=%.4f%%, slippage=%.4f%%, fixed=%.2f -> %d components",
        comm_pct, slip_pct, fixed, len(comps),
    )
    return model


def india_nse_preset() -> CompositeCostModel:
    """A realistic NSE (National Stock Exchange of India) cost preset.

    Indian equity delivery trades carry several layered charges beyond brokerage.
    Figures below are approximations of the 2024 schedule for delivery (CNC)
    trades; they are intentionally conservative (round-numbered) so the backtest
    errs on the side of HIGHER cost (under-stating alpha is always safer than
    over-stating it). Replace with a broker's exact schedule before live trading.

    Components (delivery / CNC):
      - Brokerage:       0.03% of notional (typical full-service; discount is ~Rs20/trade)
      - STT:             0.1% on BOTH legs (Securities Transaction Tax - the big one)
      - Exchange txn:    0.00345% (NSE transaction charge)
      - SEBI turnover:   Rs 10 per crore = 0.000001%
      - GST:             18% on (brokerage + exchange + SEBI) - approximated as 0.006% of notional
      - Stamp duty:      0.015% on buy only
      - Slippage:        0.05% (market-impact estimate)

    The total round-trip comes to ~0.38% (STT alone is 0.1% on EACH leg, so 0.2%
    of the round-trip is STT before anything else). A number to keep in mind when
    judging any strategy that trades frequently - and intentionally conservative
    (real slippage for liquid large-caps is lower, but erring high is safer).
    """
    return CompositeCostModel(components=[
        PercentageCost(rate=0.0003, name="brokerage"),                      # 0.03%
        TaxCost(rate=0.001, side="both", name="STT"),                       # 0.1% both legs
        PercentageCost(rate=0.0000345, name="exchange_txn"),                # 0.00345%
        TaxCost(rate=0.006 / 100, side="buy", name="stamp_duty"),           # 0.015% buy
        PercentageCost(rate=0.00006, name="GST_regulatory"),                # ~0.006% blended
        SlippageCost(pct=0.0005, name="slippage"),                          # 0.05%
    ])


# Backwards-compatible re-export so the portfolio can import CostModel/NullCostModel
# from a single canonical location in future (kept in portfolio.py for now to
# avoid touching Lesson 5's verified import graph; this module is the richer set).
