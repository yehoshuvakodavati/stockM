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

import numpy as np
import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.risk_management import RiskManager
from backtesting.strategy import Strategy
from backtesting.trade import Trade, TradeAction

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

        Timing contract (the engine's invariant, enforced here):
            At bar t, in order:
                1. mark open positions to t's price            (mark-to-market)
                2. ask RiskManager for forced exits at t's price (stops checked)
                3. execute any forced exits at t's price
                4. ask the Strategy for a signal using data <= t (no look-ahead)
                5. gate the signal through the RiskManager      (risk limits)
                6. execute the (possibly vetoed/reshaped) signal at t's price
        Decisions use information <= t; execution is at t's price (Adj Close).
        Exits (steps 2-3) come before entries (steps 4-6) so capital freed by a
        stop-out is available for a new entry the SAME bar.

        Args:
            predictions: Date-indexed table of model predictions per symbol
                         (columns: predicted_return, [confidence]). May be empty
                         for price-only strategies (e.g. Buy & Hold).
            prices:       Date-indexed canonical price (Adj Close). A Series or
                          a single-column / adj_close DataFrame.

        Returns:
            A BacktestResult with equity curve, trades, metrics, and benchmark.
        """
        from backtesting.metrics import compute_metrics
        from backtesting.strategy import _price_series

        # Normalise prices to a single Series; align predictions to its dates.
        px = _price_series(prices)
        preds = predictions if predictions is not None else pd.DataFrame(index=px.index)
        preds = preds.reindex(px.index)

        # Generate ALL signals up front (the strategy is stateless over the
        # window; it sees only data <= t by construction since predictions are
        # dated). The engine then applies them bar-by-bar, interleaving exits.
        signals = self.strategy.generate_signals(preds, px)
        sig_by_date = {s.date: s for s in signals}

        for i in range(len(px)):
            dt = px.index[i]
            price = float(px.iloc[i])
            if not np.isfinite(price) or price <= 0:
                continue
            prices_map = {self._primary_symbol(px, preds): price}
            dt_key = dt.date() if hasattr(dt, "date") else dt

            # 1) Mark to market (updates highest_since_entry for trailing stops).
            self.portfolio.mark_to_market(prices_map, dt_key)

            # 2-3) Forced exits (stops) checked AFTER mark-to-market so today's
            #      peak is current; execute them before any new entry.
            exits = self.risk_manager.check_exits(
                self.portfolio.positions, prices_map, dt_key, self.portfolio
            )
            for ex in exits:
                self.portfolio.execute(ex, price)

            # 4-6) Strategy signal, gated, executed.
            sig = sig_by_date.get(dt_key)
            if sig is not None and sig.action != TradeAction.HOLD:
                gated = self.risk_manager.apply(sig, self.portfolio)
                if gated is not None:
                    self.portfolio.execute(gated, price)

        # If no mark-to-market ran (empty prices), seed a flat curve.
        if len(self.portfolio.equity_curve) == 0:
            self.portfolio.mark_to_market(
                {self._primary_symbol(px, preds): float(px.iloc[-1]) if len(px) else 0.0},
                px.index[-1].date() if hasattr(px.index[-1], "date") else px.index[-1],
            )

        # Build a Buy & Hold benchmark on the same capital + price for comparison.
        benchmark = self._buy_and_hold_equity(px)

        metrics = compute_metrics(
            self.portfolio.equity_curve, self.portfolio.trades,
            initial_capital=self.portfolio.initial_capital,
        )
        symbols = list({s.symbol for s in signals if s.symbol}) or [self._primary_symbol(px, preds)]

        self.logger.info(
            "backtest '%s' done: %d bars, %d trades, total_return=%.2f%%",
            getattr(self.strategy, "name", "strategy"), len(px),
            len(self.portfolio.trades), metrics["total_return"] * 100,
        )
        return BacktestResult(
            strategy_name=getattr(self.strategy, "name", "strategy"),
            symbols=symbols,
            equity_curve=self.portfolio.equity_curve,
            trades=self.portfolio.trades,
            metrics=metrics,
            config=self.config,
            benchmark_equity=benchmark,
            metadata={**self.config.get("metadata", {}),
                      "n_bars": len(px), "n_trades": len(self.portfolio.trades)},
        )

    @staticmethod
    def _primary_symbol(px: pd.Series, preds: pd.DataFrame) -> str:
        """Resolve the single traded symbol (v1 is single-symbol)."""
        if preds is not None and "symbol" in getattr(preds, "columns", []):
            s = preds["symbol"].dropna()
            if len(s) > 0:
                return str(s.iloc[0])
        return getattr(px, "name", None) or "ASSET"

    def _buy_and_hold_equity(self, px: pd.Series) -> pd.Series:
        """A Buy & Hold benchmark equity curve on the same initial capital.

        Buys as many shares as possible on bar 0 with the strategy's initial
        capital and holds; no costs (a benchmark is a yardstick, not a trade).
        Returns a date-indexed Series aligned to ``px``.
        """
        if px is None or len(px) == 0:
            return pd.Series(dtype=float, name="benchmark")
        cap = self.portfolio.initial_capital
        first_price = float(px.iloc[0])
        if first_price <= 0:
            return pd.Series([cap] * len(px), index=px.index, name="benchmark")
        shares = cap / first_price
        return pd.Series(shares * px.values, index=px.index, name="benchmark")
