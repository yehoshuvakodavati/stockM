"""
StockM v1.0 - Phase 8, Lesson 7
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

The six constraints (all configurable)
---------------------------------------
    1. stop_loss_pct        - exit if price falls X% below entry (capital guard).
    2. take_profit_pct      - exit if price rises X% above entry (lock gains).
    3. trailing_stop_pct    - exit if price falls X% below the peak since entry
                              (ride winners, cut losers - the professional's
                              favourite). Uses Position.highest_since_entry,
                              which the portfolio updates every bar.
    4. max_risk_per_trade_pct- cap how much capital an ENTRY can risk (sizing).
    5. max_position_pct     - cap a single position as a fraction of equity.
    6. max_daily_loss_pct   - halt new entries after a -X% day (circuit breaker).

Dependency: imports backtesting.trade; references Portfolio only under
TYPE_CHECKING (to avoid a circular import: Portfolio owns execution, Risk
owns limits - neither should import the other at runtime).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING, Any

from backtesting.trade import Signal, TradeAction

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from backtesting.portfolio import Portfolio, Position

logger = logging.getLogger("stockm.backtesting.risk")


class RiskManager:
    """Enforce risk constraints on every signal and manage exit triggers.

    Two duties, called by the engine on every bar:
      1. ``apply(signal, portfolio)``         - gate an ENTRY signal (before execution).
      2. ``check_exits(positions, prices, dt)`` - forced EXIT signals (after execution).

    All thresholds are optional (None = rule disabled). Passing an empty config
    or all-None thresholds makes the manager a no-op pass-through - so a backtest
    can run with "no risk management" for an honest baseline, then layer rules in.

    Config keys (any subset; units are percent NUMBERS in config, fractions here):
        stop_loss_pct:         e.g. 5.0  -> exit at -5% from entry.
        take_profit_pct:       e.g. 15.0 -> exit at +15% from entry.
        trailing_stop_pct:     e.g. 10.0 -> exit at -10% from peak.
        max_risk_per_trade_pct: e.g. 2.0 -> cap entry size so a stop-out risks <= 2% of equity.
        max_position_pct:      e.g. 25.0 -> cap any position at 25% of equity.
        max_daily_loss_pct:    e.g. 5.0  -> block new entries after a -5% day.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}

        def _frac(key: str) -> float | None:
            """Read a percent NUMBER from config -> fraction. None if absent/<=0."""
            v = cfg.get(key)
            if v is None:
                return None
            f = float(v) / 100.0
            return f if f > 0 else None

        self.stop_loss_pct = _frac("stop_loss_pct")
        self.take_profit_pct = _frac("take_profit_pct")
        self.trailing_stop_pct = _frac("trailing_stop_pct")
        self.max_risk_per_trade_pct = _frac("max_risk_per_trade_pct")
        self.max_position_pct = _frac("max_position_pct")
        self.max_daily_loss_pct = _frac("max_daily_loss_pct")

        # Circuit-breaker state: the equity at the start of the current day,
        # used to detect a max_daily_loss breach. Updated by check_exits each bar.
        self._day_start_equity: float | None = None
        self._current_day: date | None = None
        # When True, new BUYs are blocked until the next trading day.
        self._halted = False

        active = [k for k, v in [
            ("stop_loss", self.stop_loss_pct), ("take_profit", self.take_profit_pct),
            ("trailing_stop", self.trailing_stop_pct), ("max_risk/trade", self.max_risk_per_trade_pct),
            ("max_position", self.max_position_pct), ("max_daily_loss", self.max_daily_loss_pct),
        ] if v is not None]
        logger.info("RiskManager active rules: %s", active or ["none (pass-through)"])

    # ------------------------------------------------------------------ gates

    def apply(self, signal: Signal, portfolio: "Portfolio") -> Signal | None:
        """Gate an ENTRY signal: return it (possibly reshaped) or None to veto.

        Called by the engine BEFORE ``portfolio.execute``. A vetoed BUY never
        reaches the portfolio; a reshaped BUY has its ``strength`` (sizing
        fraction) capped so the position stays within limits.

        Args:
            signal:    The strategy's intended action (typically a BUY entry).
            portfolio: Current portfolio (for equity + existing position size).

        Returns:
            The (possibly reshaped) Signal, or None to veto.
        """
        if signal.action != TradeAction.BUY:
            # SELLs are exits - risk limits apply to ENTRIES. Exits flow through
            # to the portfolio (which no-ops a SELL with no position). The
            # *forced* exits (stops) are generated separately in check_exits.
            return signal

        # --- Circuit breaker: block new entries after a max-daily-loss breach.
        if self._halted:
            logger.debug("veto BUY %s: daily-loss halt active", signal.symbol)
            return None

        # --- No position price needed for the equity-fraction caps; we need a
        #     reference equity. The portfolio exposes total_equity(prices), but
        #     apply() is called before mark_to_market, so we use cash + held
        #     value at the LAST known prices. For sizing caps, current equity
        #     is a good enough proxy (the engine passes the fill price via the
        #     signal's companion context in a fuller design; here we cap by
        #     the portfolio's cash, which is the binding constraint anyway).
        equity = self._approximate_equity(portfolio)

        # --- max_position_pct: cap the position size as a fraction of equity.
        strength = float(signal.strength)
        if self.max_position_pct is not None and equity > 0:
            # If an existing position + the proposed buy would exceed the cap,
            # shrink the buy's strength. v1 sizing deploys equity*strength, so
            # capping strength directly caps the deployed capital.
            existing = self._existing_position_value(portfolio, signal.symbol)
            allowed_notional = (equity * self.max_position_pct) - existing
            max_deployable = max(allowed_notional, 0.0)
            # strength maps to fraction of equity deployed; cap it.
            strength = min(strength, max_deployable / equity) if equity > 0 else 0.0
            if strength <= 0:
                logger.debug("veto BUY %s: max_position_pct cap reached", signal.symbol)
                return None

        # --- max_risk_per_trade_pct: cap size so a stop-out risks <= X% of equity.
        if self.max_risk_per_trade_pct is not None and self.stop_loss_pct is not None and equity > 0:
            # Risk per share ~= entry_price * stop_loss_pct. The total $ at risk
            # on the new buy = qty * risk_per_share = (deploy/equity)*equity... so
            # we want: deploy * stop_loss_pct <= equity * max_risk_per_trade_pct
            # => strength (= deploy/equity) <= max_risk_per_trade_pct / stop_loss_pct
            max_strength_by_risk = self.max_risk_per_trade_pct / self.stop_loss_pct
            strength = min(strength, max_strength_by_risk)
            if strength <= 0:
                logger.debug("veto BUY %s: max_risk_per_trade_pct cap", signal.symbol)
                return None

        # Clamp to [0, 1] and reshape if we actually capped it.
        strength = max(0.0, min(strength, 1.0))
        if strength < float(signal.strength):
            logger.debug("reshape BUY %s strength %.3f -> %.3f", signal.symbol, signal.strength, strength)
            return replace(signal, strength=strength, reason=signal.reason + "+risk:capped")
        return signal

    def check_exits(
        self, positions: dict, prices: dict[str, float], dt: date,
        portfolio: "Portfolio | None" = None,
    ) -> list[Signal]:
        """Generate forced exit signals for stops / triggers hit this bar.

        Called by the engine AFTER execution and mark_to_market. Scans every
        OPEN position and emits a SELL signal if any exit trigger fires:

            - stop_loss:      price <= entry * (1 - stop_loss_pct)
            - take_profit:    price >= entry * (1 + take_profit_pct)
            - trailing_stop:  price <= peak * (1 - trailing_stop_pct)

        Also updates the daily-loss circuit-breaker state.

        Args:
            positions: {symbol: Position} from the portfolio.
            prices:    {symbol: current_price} for this bar.
            dt:        The bar's date.
            portfolio: Optional portfolio for true mark-to-market equity in the
                       daily-loss breaker. If None, the breaker uses held-value
                       at mark prices as a proxy (unit-test friendly).

        Returns:
            List of forced SELL Signals (strength=1.0 = full exit). Empty if none.
        """
        self._update_day_breaker(positions, prices, dt, portfolio)

        exits: list[Signal] = []
        for sym, pos in positions.items():
            if not getattr(pos, "is_open", False):
                continue
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            entry = pos.avg_entry_price
            peak = pos.highest_since_entry or entry

            reasons: list[str] = []
            if self.stop_loss_pct is not None and price <= entry * (1 - self.stop_loss_pct):
                reasons.append(f"stop_loss@{self.stop_loss_pct:.1%}")
            if self.take_profit_pct is not None and price >= entry * (1 + self.take_profit_pct):
                reasons.append(f"take_profit@{self.take_profit_pct:.1%}")
            if (self.trailing_stop_pct is not None
                    and peak > 0 and price <= peak * (1 - self.trailing_stop_pct)):
                reasons.append(f"trailing_stop@{self.trailing_stop_pct:.1%}")

            if reasons:
                exits.append(Signal(
                    date=dt, symbol=sym, action=TradeAction.SELL,
                    strength=1.0,  # full exit
                    reason="risk:" + "+".join(reasons),
                ))
        if exits:
            logger.debug("check_exits %s: %d forced exits", dt, len(exits))
        return exits

    # ------------------------------------------------------------- internals

    def _approximate_equity(self, portfolio: "Portfolio") -> float:
        """Best-effort current equity for sizing caps (cash + held value).

        ``apply`` runs before mark_to_market, so we approximate equity as cash
        plus the held positions valued at their average entry (a conservative
        lower-bound proxy when live mark prices aren't available yet). The
        portfolio's ``total_equity`` needs current prices; this avoids needing
        them at gate time. The cap is a guard rail, not a precision instrument.
        """
        cash = portfolio.cash
        held = 0.0
        for pos in portfolio.positions.values():
            if pos.is_open:
                held += pos.quantity * pos.avg_entry_price
        return cash + held

    def _existing_position_value(self, portfolio: "Portfolio", symbol: str) -> float:
        """Notional of any currently-open position in ``symbol`` (at entry cost)."""
        pos = portfolio.positions.get(symbol)
        if pos is None or not pos.is_open:
            return 0.0
        return pos.quantity * pos.avg_entry_price

    def _update_day_breaker(
        self, positions: dict, prices: dict[str, float], dt: date,
        portfolio: "Portfolio | None" = None,
    ) -> None:
        """Track the daily-loss circuit breaker.

        On the first bar of a new trading day, record the starting equity. On
        every bar, check if equity has fallen more than ``max_daily_loss_pct``
        from the day's start; if so, set the halt flag (blocks new BUYs in
        ``apply`` until the next day). The halt resets at the start of each new
        day, giving the strategy a fresh allowance.
        """
        if self.max_daily_loss_pct is None:
            return
        if dt != self._current_day:
            # New trading day: reset the halt and record the start-of-day equity.
            self._current_day = dt
            self._halted = False
            self._day_start_equity = self._equity_for_breaker(positions, prices, portfolio)
            return
        if self._day_start_equity is None or self._day_start_equity <= 0:
            return
        current = self._equity_for_breaker(positions, prices, portfolio)
        drawdown = (self._day_start_equity - current) / self._day_start_equity
        if drawdown >= self.max_daily_loss_pct and not self._halted:
            self._halted = True
            logger.warning(
                "max_daily_loss halt triggered %s: drawdown %.2f%% >= %.2f%%",
                dt, drawdown * 100, self.max_daily_loss_pct * 100,
            )

    def _equity_for_breaker(
        self, positions: dict, prices: dict[str, float],
        portfolio: "Portfolio | None" = None,
    ) -> float:
        """Mark-to-market equity for the daily-loss breaker.

        Prefers the portfolio's true ``total_equity`` (cash + mark value) when
        the engine passes it; falls back to held-value at mark prices for the
        standalone (positions, prices) contract used in unit tests. The breaker
        must reflect MARK losses, so we never use entry-cost here.
        """
        if portfolio is not None:
            try:
                return float(portfolio.total_equity(prices))
            except Exception:  # noqa: BLE001 - fall back to proxy on any error
                pass
        held = 0.0
        for sym, pos in positions.items():
            if getattr(pos, "is_open", False) and sym in prices:
                held += pos.quantity * prices[sym]
        return held
