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

import numpy as np
import pandas as pd

from backtesting.trade import Trade, TradeAction

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


def _daily_returns(equity_curve: pd.Series) -> pd.Series:
    """Pct change of the equity curve. The input to Sharpe/Sortino.

    The equity curve is recorded once per bar (daily), so pct change gives
    daily simple returns. A flat/short curve yields an empty Series, which the
    callers handle (ratios return 0.0 rather than NaN).
    """
    if equity_curve is None or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    rets = equity_curve.pct_change().dropna()
    # Guard against zero-equity (a bankrupt curve) producing inf.
    return rets.replace([np.inf, -np.inf], np.nan).dropna()


def total_return(equity_curve: pd.Series) -> float:
    """Total return over the whole backtest.

    total_return = equity_final / equity_initial - 1

    Uses the equity curve (not trade-by-trade), so it captures both realised
    and unrealised P&L as of the last bar.
    """
    if equity_curve is None or len(equity_curve) < 1:
        return 0.0
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    if start <= 0:
        return 0.0
    return end / start - 1.0


def cagr(equity_curve: pd.Series, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound Annual Growth Rate.

        CAGR = (equity_final / equity_initial) ^ (252 / n_days) - 1

    Annualises the total return over the number of trading days elapsed, so
    a 14.5% return over 3 years (~756 days) is a ~4.6% CAGR. This is the
    "interest rate" the strategy earned, compounded annually.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    n_days = len(equity_curve) - 1  # periods, not points
    if start <= 0 or end <= 0 or n_days <= 0:
        return 0.0
    return (end / start) ** (trading_days_per_year / n_days) - 1.0


def sharpe_ratio(
    equity_curve: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio: risk-adjusted return per unit of TOTAL volatility.

        Sharpe = (mean_daily_ret - rf_daily) / std_daily_ret * sqrt(252)

    The Sharpe penalises BOTH upside and downside volatility (std uses all
    deviations). A high Sharpe means smooth, consistent returns; a low/negative
    Sharpe means volatile or losing. Annualised by sqrt(252) because variance
    scales linearly with time, std with sqrt(time). rf is the daily risk-free.
    """
    rets = _daily_returns(equity_curve)
    if len(rets) < 2:
        return 0.0
    rf_daily = risk_free_rate / trading_days_per_year
    excess = rets - rf_daily
    std = float(rets.std(ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(trading_days_per_year))


def sortino_ratio(
    equity_curve: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sortino ratio: return per unit of DOWNSIDE volatility only.

        Sortino = (mean_daily_ret - rf_daily) / downside_std * sqrt(252)

    The Sortino fixes the Sharpe's key flaw: it does NOT penalise upside
    volatility (large positive returns inflate Sharpe's std but are good).
    downside_std counts only returns below the target (rf). A strategy with
    occasional large wins but small, rare losses has a much higher Sortino
    than Sharpe - which is the honest picture for a trend-follower.
    """
    rets = _daily_returns(equity_curve)
    if len(rets) < 2:
        return 0.0
    rf_daily = risk_free_rate / trading_days_per_year
    excess = rets - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        # No downside deviations - infinite risk-adjusted return. Cap at a
        # large finite value rather than inf so downstream comparison works.
        return 0.0 if float(excess.mean()) <= 0 else 99.0
    downside_std = float(np.sqrt((downside ** 2).mean()))  # target semideviation
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(trading_days_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown: the worst peak-to-trough loss on the equity curve.

    A drawdown is the % drop from a running maximum to a subsequent trough.
    max_drawdown is the deepest such drop over the whole backtest - the "how
    much could you lose at the worst moment" metric. Returned as a NEGATIVE
    number (e.g. -0.25 = -25%). Risk-averse investors care about this more
    than return: a strategy that returns 20% but draws down 60% is often
    uninvestable (you'd capitulate before recovery).
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    curve = equity_curve.astype(float)
    running_max = curve.cummax()
    drawdowns = (curve - running_max) / running_max  # <= 0 everywhere
    mdd = float(drawdowns.min())
    if np.isnan(mdd):
        return 0.0
    return mdd


def calmar_ratio(
    equity_curve: pd.Series,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio: CAGR per unit of maximum drawdown.

        Calmar = CAGR / |max_drawdown|

    Return per unit of worst-case pain. A Calmar > 1 means the annualised
    return exceeds the worst drawdown (good); < 1 means drawdowns exceed
    annual return (rough ride). Unlike Sharpe (daily volatility), Calmar
    focuses on the SINGLE worst event, suiting investors who fear ruin more
    than variance.
    """
    mdd = max_drawdown(equity_curve)
    if mdd == 0:  # no drawdown -> no pain; cap to avoid division-by-zero
        c = cagr(equity_curve, trading_days_per_year)
        return c if c > 0 else 0.0
    return cagr(equity_curve, trading_days_per_year) / abs(mdd)


def _trade_pnl(trades: list[Trade], initial_capital: float | None = None) -> list[float]:
    """Realised P&L per ROUND-TRIP trade (buy + matching sell).

    A position's P&L is only known when it CLOSES. We pair buys with sells
    (FIFO) and compute each round-trip's profit = sell_proceeds - buy_cost -
    costs. Partial fills are handled by tracking the remaining quantity from
    each buy lot. This gives the per-trade win/loss list for win_rate,
    profit_factor, and expectancy.

    If ``initial_capital`` is None we can't infer it; the round-trip P&L is
    independent of capital (it's per-share), so this is fine for the ratios.
    """
    if not trades:
        return []
    # Build FIFO buy lots: (price, qty_remaining, cost_incurred).
    buy_lots: list[list[float]] = []  # [price, qty, cost]
    pnls: list[float] = []
    for t in trades:
        if t.action == TradeAction.BUY:
            buy_lots.append([t.price, t.quantity, t.cost])
        elif t.action == TradeAction.SELL:
            qty_to_fill = t.quantity
            sell_proceeds_net = (t.price * t.quantity) - t.cost  # net of sell cost
            # Match against FIFO lots until the sell is fully allocated.
            lot_proceeds = 0.0
            lot_costs = 0.0
            remaining_sell = qty_to_fill
            while remaining_sell > 1e-12 and buy_lots:
                lot = buy_lots[0]
                take = min(lot[1], remaining_sell)
                frac = take / t.quantity if t.quantity > 0 else 0
                lot_proceeds += frac * sell_proceeds_net
                lot_costs += (lot[0] * take) + (frac * lot[2])  # buy cost + proportional buy fee
                lot[1] -= take
                remaining_sell -= take
                if lot[1] <= 1e-12:
                    buy_lots.pop(0)
            if qty_to_fill > 0:
                pnls.append(lot_proceeds - lot_costs)
    return pnls


def win_rate(trades: list[Trade]) -> float:
    """Fraction of round-trip trades that were profitable.

    win_rate = n_winning_trades / n_round_trips

    A 55% win rate sounds great but is meaningless alone: a strategy that wins
    55% of the time by ₹1 and loses 45% of the time by ₹10 is a money loser.
    Read alongside profit_factor and expectancy.
    """
    pnls = _trade_pnl(trades)
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def profit_factor(trades: list[Trade]) -> float:
    """Profit factor: gross profit / gross loss.

        PF = sum(wins) / |sum(losses)|

    PF > 1 = profitable; PF > 1.5 = solid; PF > 2 = excellent. Unlike win
    rate, it weights by SIZE, so it captures "win small, lose big" pathologies.
    A 40%-win-rate strategy with a PF of 2 (rare big wins, frequent small
    losses - a classic trend-follower) is excellent; a 70%-win-rate strategy
    with PF 0.8 is a slow bleed.
    """
    pnls = _trade_pnl(trades)
    if not pnls:
        return 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)  # positive number
    if gross_loss == 0:
        return 99.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_profit(trades: list[Trade]) -> float:
    """Average ₹ amount of a WINNING round-trip trade."""
    pnls = _trade_pnl(trades)
    wins = [p for p in pnls if p > 0]
    if not wins:
        return 0.0
    return float(np.mean(wins))


def avg_loss(trades: list[Trade]) -> float:
    """Average ₹ amount of a LOSING round-trip trade (negative number)."""
    pnls = _trade_pnl(trades)
    losses = [p for p in pnls if p < 0]
    if not losses:
        return 0.0
    return float(np.mean(losses))


def expectancy(trades: list[Trade]) -> float:
    """Expected ₹ profit per round-trip trade.

        E = win_rate * avg_profit + (1 - win_rate) * avg_loss

    The single most important trading metric: it's the expected value of one
    trade. Positive expectancy = a winning system (in expectation); negative =
    a losing one. A strategy with 40% win rate, avg profit ₹300, avg loss ₹100
    has expectancy 0.4*300 + 0.6*(-100) = ₹60/trade - profitable despite
    losing more often than winning.
    """
    pnls = _trade_pnl(trades)
    if not pnls:
        return 0.0
    wr = win_rate(trades)
    ap = avg_profit(trades)
    al = avg_loss(trades)
    return wr * ap + (1 - wr) * al


def turnover(equity_curve: pd.Series, trades: list[Trade], initial_capital: float) -> float:
    """Average annual turnover: traded value per year / capital.

        turnover = sum(|trade_notional|) / n_years / initial_capital

    Measures how much the strategy trades. High turnover = high cost drag
    (Lesson 6) and high operational burden. A buy-and-hold has turnover ~2
    (one buy + one sell over the period, annualised tiny); a daily trader can
    hit 100+ (trading its whole book many times a year). This connects directly
    to the cost-drag arithmetic from Lesson 6.
    """
    if not trades or initial_capital <= 0 or equity_curve is None or len(equity_curve) < 2:
        return 0.0
    n_years = (len(equity_curve) - 1) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    total_notional = sum(float(t.value) for t in trades)
    return total_notional / n_years / initial_capital


def n_trades(trades: list[Trade]) -> float:
    """Number of executed trades (BUY + SELL fills). As float for the dict."""
    return float(len(trades))


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[Trade],
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    initial_capital: float | None = None,
) -> dict[str, float]:
    """Compute the full metric suite from an equity curve + trade log.

    Each component is a standalone pure function above; this composes them into
    the flat dict with every key in :data:`METRIC_NAMES`. Lesson 10's strategy
    comparison and Lesson 12's report both rely on these exact keys existing.

    Args:
        equity_curve:          Date-indexed Series of total portfolio value.
        trades:                Executed Trades from the portfolio.
        trading_days_per_year: 252 (NSE) default; override for crypto/weekly.
        risk_free_rate:        Annualised risk-free rate (default 0).
        initial_capital:       Starting capital (for turnover). If None,
                               inferred from the equity curve's first value.

    Returns:
        Dict keyed by METRIC_NAMES. All values are floats; ratios return 0.0
        (never NaN/inf) so downstream comparisons and JSON serialisation work.
    """
    cap = initial_capital if initial_capital is not None else (
        float(equity_curve.iloc[0]) if equity_curve is not None and len(equity_curve) > 0 else 0.0
    )
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve, trading_days_per_year),
        "sharpe_ratio": sharpe_ratio(equity_curve, trading_days_per_year, risk_free_rate),
        "sortino_ratio": sortino_ratio(equity_curve, trading_days_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve, trading_days_per_year),
        "profit_factor": profit_factor(trades),
        "win_rate": win_rate(trades),
        "avg_profit": avg_profit(trades),
        "avg_loss": avg_loss(trades),
        "expectancy": expectancy(trades),
        "turnover": turnover(equity_curve, trades, cap),
        "n_trades": n_trades(trades),
    }
