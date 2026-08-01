"""
StockM v1.0 - Phase 8, Lesson 10
Strategy Comparison Framework
==============================

Single Responsibility: run multiple strategies through the SAME backtest
(predictions, prices, costs, risk regime) and present them side-by-side on
return, risk, stability, consistency, and drawdown - then recommend the
strongest.

Why a dedicated comparison layer
--------------------------------
A single backtest answers "how did THIS strategy do?". A comparison answers
"which strategy should we deploy?". That requires running every candidate on
identical data + costs + risk so the only variable is the strategy itself
(apples-to-apples). Lesson 8's pure metrics make this possible: the SAME
compute_metrics scores every run. This module orchestrates N runs and ranks
them, plus a RandomStrategy baseline (the "no-skill" floor) and Buy & Hold
(the "no-model" benchmark).

The honest mandate
------------------
Given ~50% directional accuracy (see dl-baseline-findings), we EXPECT most
prediction strategies to lose to Buy & Hold after costs. The comparison must
SURFACE that, not hide it: the recommendation is "deploy the strategy that
loses least / wins most under the regime you'll trade in", and if none beats
Buy & Hold, the honest recommendation is Buy & Hold. A framework that
recommended a cost-losing strategy would be malpractice.

Scoring
-------
Strategies are ranked by a composite score balancing return AND risk:
    score = 0.45 * norm(total_return)
          + 0.30 * norm(sharpe_ratio)
          + 0.15 * (1 - norm(|max_drawdown|))     # shallower drawdown is better
          + 0.10 * norm(profit_factor capped at 3) # reward profitability
Each axis is min-max normalised across the candidates so no single metric
dominates by scale. Weights favour return (you deploy to make money) but
penalise risk (you survive to keep it). The top scorer is "recommended"; the
report notes whether it beat Buy & Hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.portfolio import Portfolio
from backtesting.risk_management import RiskManager
from backtesting.strategy import Strategy, BuyAndHoldStrategy, get_strategy, register_strategy
from backtesting.trade import TradeAction

logger = logging.getLogger("stockm.backtesting.comparison")

# Metric weights for the composite score (sum to 1.0).
SCORE_WEIGHTS = {"total_return": 0.45, "sharpe_ratio": 0.30,
                 "max_drawdown": 0.15, "profit_factor": 0.10}


@dataclass
class ComparisonRow:
    """One strategy's summary in the comparison table."""

    strategy_name: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    expectancy: float
    turnover: float
    n_trades: int
    beats_benchmark: bool
    score: float = 0.0  # composite score, filled by the ranker


@dataclass
class ComparisonResult:
    """The full output of a strategy comparison."""

    rows: list[ComparisonRow]  # ranked by score (best first)
    benchmark_return: float
    recommended: str  # strategy_name of the top scorer
    recommended_beats_benchmark: bool
    results: list[BacktestResult] = field(default_factory=list)  # full results for charts

    def to_dataframe(self) -> pd.DataFrame:
        """A ranked summary table (for the report / console)."""
        return pd.DataFrame([{
            "strategy": r.strategy_name,
            "return": r.total_return,
            "cagr": r.cagr,
            "sharpe": r.sharpe_ratio,
            "sortino": r.sortino_ratio,
            "max_dd": r.max_drawdown,
            "calmar": r.calmar_ratio,
            "profit_factor": r.profit_factor,
            "win_rate": r.win_rate,
            "expectancy": r.expectancy,
            "turnover": r.turnover,
            "n_trades": r.n_trades,
            "beats_bm": r.beats_benchmark,
            "score": r.score,
        } for r in self.rows])


def _norm(values: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]; all-equal -> 0.5 (no discrimination)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class StrategyComparator:
    """Run N strategies on the same data and rank them.

    Args:
        initial_capital: Starting cash for every run (identical conditions).
        cost_model:      Transaction cost model (Lesson 6). Same for all runs.
        risk_config:     Risk-manager config (Lesson 7). Same for all runs.
        benchmark_name:  The strategy to treat as the "beat this" benchmark
                         (default "buy_and_hold").
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        cost_model: Any | None = None,
        risk_config: dict | None = None,
        benchmark_name: str = "buy_and_hold",
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_model = cost_model
        self.risk_config = risk_config or {}
        self.benchmark_name = benchmark_name

    def run(
        self,
        strategies: list[Strategy],
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> ComparisonResult:
        """Run every strategy through the same backtest; rank and recommend.

        Args:
            strategies: The candidate strategies to compare.
            predictions: Date-indexed model predictions (predicted_return).
            prices:      Date-indexed canonical price (Adj Close).

        Returns:
            A ComparisonResult with ranked rows, the recommended strategy, and
            the full BacktestResults for downstream charts.
        """
        results: list[BacktestResult] = []
        for strat in strategies:
            pf = Portfolio(self.initial_capital, cost_model=self.cost_model)
            rm = RiskManager(self.risk_config)
            eng = BacktestEngine(strategy=strat, portfolio=pf, risk_manager=rm,
                                 config={"initial_capital": self.initial_capital})
            try:
                res = eng.run(predictions, prices)
                results.append(res)
            except Exception as e:  # noqa: BLE001 - one failure must not sink the comparison
                logger.error("strategy '%s' failed: %s", getattr(strat, "name", "?"), e)

        rows = [self._row(r) for r in results]
        self._score(rows)
        rows.sort(key=lambda r: r.score, reverse=True)

        benchmark_return = self._benchmark_return(results)
        recommended = rows[0].strategy_name if rows else ""
        recommended_beats = rows[0].beats_benchmark if rows else False

        logger.info(
            "comparison: %d strategies, recommended='%s' (beats benchmark=%s)",
            len(rows), recommended, recommended_beats,
        )
        return ComparisonResult(
            rows=rows, benchmark_return=benchmark_return,
            recommended=recommended, recommended_beats_benchmark=recommended_beats,
            results=results,
        )

    # ----------------------------------------------------------- helpers

    def _row(self, result: BacktestResult) -> ComparisonRow:
        """Build a ComparisonRow from a BacktestResult's metrics."""
        m = result.metrics
        bm_ret = (result.benchmark_equity.iloc[-1] / result.benchmark_equity.iloc[0] - 1
                  if result.benchmark_equity is not None and len(result.benchmark_equity) else 0.0)
        return ComparisonRow(
            strategy_name=result.strategy_name,
            total_return=m["total_return"],
            cagr=m["cagr"],
            sharpe_ratio=m["sharpe_ratio"],
            sortino_ratio=m["sortino_ratio"],
            max_drawdown=m["max_drawdown"],
            calmar_ratio=m["calmar_ratio"],
            profit_factor=min(m["profit_factor"], 3.0),  # cap for scoring
            win_rate=m["win_rate"],
            expectancy=m["expectancy"],
            turnover=m["turnover"],
            n_trades=int(m["n_trades"]),
            beats_benchmark=m["total_return"] > bm_ret,
        )

    def _score(self, rows: list[ComparisonRow]) -> None:
        """Fill the composite score (min-max normalised, weighted)."""
        if not rows:
            return
        rets = _norm([r.total_return for r in rows])
        shrs = _norm([r.sharpe_ratio for r in rows])
        # Drawdown: shallower (closer to 0) is better -> normalise |mdd|, invert.
        dds = _norm([abs(r.max_drawdown) for r in rows])
        pfs = _norm([r.profit_factor for r in rows])
        for i, r in enumerate(rows):
            r.score = (SCORE_WEIGHTS["total_return"] * rets[i]
                       + SCORE_WEIGHTS["sharpe_ratio"] * shrs[i]
                       + SCORE_WEIGHTS["max_drawdown"] * (1.0 - dds[i])
                       + SCORE_WEIGHTS["profit_factor"] * pfs[i])

    def _benchmark_return(self, results: list[BacktestResult]) -> float:
        """The benchmark strategy's total return (for the 'beat this' line)."""
        for r in results:
            if r.strategy_name == self.benchmark_name:
                return r.metrics["total_return"]
        # Fallback: the buy-and-hold equity embedded in any result.
        for r in results:
            if r.benchmark_equity is not None and len(r.benchmark_equity):
                return float(r.benchmark_equity.iloc[-1] / r.benchmark_equity.iloc[0] - 1)
        return 0.0


# ---------------------------------------------------------------------------
# Standard comparison suite (Lesson 10's required candidates)
# ---------------------------------------------------------------------------

def standard_suite(
    symbol: str,
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    cost_model: Any | None = None,
    risk_config: dict | None = None,
) -> ComparisonResult:
    """Run the Lesson 10 required candidate set on one symbol.

    Candidates:
        - buy_and_hold            (the benchmark)
        - random                  (the no-skill floor)
        - prediction_based        (pure-direction model strategy)
        - threshold               (noise-filtered model strategy)
        - ma_confirmation         (model + trend filter)

    For a real model comparison, replace the synthetic ``predictions`` with the
    ML/DL predictor output (models.prediction.predict / deep_learning.inference).
    The candidates here use whatever predictions are passed, so the comparison
    is honest about the signal quality fed in.
    """
    from backtesting.portfolio import NullCostModel

    cm = cost_model if cost_model is not None else NullCostModel()
    candidates: list[Strategy] = [
        BuyAndHoldStrategy(symbol=symbol),
        RandomStrategy(symbol=symbol, seed=42),
        get_strategy("prediction_based", threshold=0.0, symbol=symbol),
        get_strategy("threshold", threshold=0.002, symbol=symbol),
        get_strategy("ma_confirmation", ma_window=20, symbol=symbol),
    ]
    comp = StrategyComparator(initial_capital=initial_capital, cost_model=cm,
                              risk_config=risk_config)
    return comp.run(candidates, predictions, prices)


@register_strategy
class RandomStrategy(Strategy):
    """A no-skill floor: BUY/SELL at random. The "is the model better than
    chance?" baseline.

    If a prediction strategy can't beat Random after costs, the model adds no
    value beyond noise. Uses a seeded RNG for reproducibility (Lesson 13).
    """

    name = "random"

    def __init__(self, symbol: str = "", seed: int = 42, hold_prob: float = 0.0) -> None:
        self.symbol = symbol
        self.seed = seed
        self.hold_prob = hold_prob  # fraction of bars to HOLD (0 = trade every bar)

    def generate_signals(self, predictions: pd.DataFrame, prices: pd.DataFrame) -> list:
        from backtesting.trade import Signal
        rng = np.random.default_rng(self.seed)
        idx = predictions.index if predictions is not None and len(predictions) else prices.index
        signals = []
        for ts in idx:
            roll = rng.random()
            if roll < self.hold_prob:
                action = TradeAction.HOLD
            elif roll < 0.5 + self.hold_prob / 2:
                action = TradeAction.BUY
            else:
                action = TradeAction.SELL
            d = ts.date() if hasattr(ts, "date") else ts
            signals.append(Signal(date=d, symbol=self.symbol, action=action, reason="random"))
        return signals
