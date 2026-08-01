"""
StockM v1.0 - Phase 8, Lesson 11
Stress Testing & Regime Analysis
================================

Single Responsibility: slice the price history into market REGIMES (bull, bear,
sideways, high/low volatility) and re-run the strategy comparison in each - so
we know how a strategy behaves when conditions change, not just on average.

Why stress test by regime
-------------------------
A strategy's average return hides regime dependence. A trend-follower that
makes +40% in bull markets but loses -30% in sideways ones looks fine on
average (+10%) but is uninvestable if you can't predict the regime. Equally, a
mean-reverter might shine in sideways markets and die in trends. Stress
testing slices the backtest into regimes and scores each separately, exposing
WHEN a strategy works and when it fails - the robustness question.

Professionals care about this because live markets cycle through regimes. A
strategy deployed after a bull-market backtest will eventually meet a bear
market; if it has never been tested in one, the first drawdown is a surprise
and surprises cause capitulation. Stress testing converts "average performance"
into "performance in each condition I'll actually face".

Regime detection (deterministic, look-ahead-safe)
-------------------------------------------------
Regimes are classified by a TRAILING 200-day moving-average slope (trend) and
trailing 63-day realised volatility (vol bucket). Both use only data <= t, so
the segmentation never peeks into the future. A bar's regime is fixed by its
own trailing window, then we slice contiguous runs of the same regime into
sub-periods for per-regime backtests.

    trend:  200d-MA 63d slope > +threshold  -> bull
                              < -threshold  -> bear
                              otherwise     -> sideways
    vol:    63d annualised vol > median+0.5*std -> high_vol
                              < median-0.5*std -> low_vol
                              otherwise        -> normal_vol

Dependency: imports backtesting.comparison (StrategyComparator) + engine/trade.
Reuses the SAME metrics + engine so per-regime results are apples-to-apples
with the full-period comparison (Lesson 10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtesting.comparison import ComparisonResult, StrategyComparator
from backtesting.strategy import Strategy

logger = logging.getLogger("stockm.backtesting.stress_testing")

# Default regime-detection params (trailing windows -> look-ahead-safe).
TREND_MA_WINDOW = 200
TREND_SLOPE_WINDOW = 63
TREND_THRESHOLD = 0.02       # 200d-MA 63d slope magnitude for bull/bear
VOL_WINDOW = 63              # ~3-month realised vol
VOL_HIGH_QUANTILE = 0.75     # top quartile = high-vol regime
VOL_LOW_QUANTILE = 0.25      # bottom quartile = low-vol regime

REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"


@dataclass
class RegimeSlice:
    """A contiguous sub-period of a single market regime.

    Attributes:
        regime:    "bull" / "bear" / "sideways" (trend) or "high_vol" /
                   "low_vol" / "normal_vol" (vol bucket).
        start:     First date (inclusive).
        end:       Last date (inclusive).
        n_bars:    Number of trading bars in the slice.
        ann_return: Annualised buy-and-hold return over the slice (regime flavour).
    """

    regime: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_bars: int
    ann_return: float


@dataclass
class RegimeResult:
    """One regime's comparison outcome."""

    regime: str
    slices: list[RegimeSlice]
    comparison: ComparisonResult
    n_bars: int


@dataclass
class StressTestResult:
    """The full cross-regime stress test."""

    by_regime: dict[str, RegimeResult] = field(default_factory=dict)
    strategy_names: list[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        """A wide table: rows = strategies, columns = regime returns.

        The single most useful stress-test view: how does each strategy's
        return change across regimes? A row that's positive everywhere is
        robust; one that flips sign is regime-dependent (fragile).
        """
        rows = []
        regimes = sorted(self.by_regime.keys())
        for name in self.strategy_names:
            row = {"strategy": name}
            for reg in regimes:
                rr = self.by_regime[reg]
                # Find this strategy's row in the regime's comparison.
                match = [r for r in rr.comparison.rows if r.strategy_name == name]
                row[reg] = match[0].total_return if match else np.nan
            rows.append(row)
        return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# Regime classification (pure functions over a price Series)
# ---------------------------------------------------------------------------

def classify_trend_regimes(
    prices: pd.Series,
    ma_window: int = TREND_MA_WINDOW,
    slope_window: int = TREND_SLOPE_WINDOW,
    threshold: float = TREND_THRESHOLD,
) -> pd.Series:
    """Classify each bar as bull / bear / sideways by trailing 200d-MA slope.

    All look-ahead-safe: the MA and its slope at bar t use only data <= t.

    Args:
        prices:       Date-indexed canonical price (Adj Close).
        ma_window:    Trailing MA length (default 200).
        slope_window: Over how many bars to measure the MA's slope (default 63).
        threshold:    |slope| above this => bull or bear; else sideways.

    Returns:
        Date-indexed Series of regime labels ("bull"/"bear"/"sideways"). Bars
        before the MA/slope are warmup -> NaN (excluded from slices).
    """
    px = prices.astype(float)
    ma = px.rolling(ma_window, min_periods=ma_window).mean()
    slope = ma.pct_change(slope_window)
    regime = pd.Series("sideways", index=px.index, dtype=object)
    regime[slope > threshold] = REGIME_BULL
    regime[slope < -threshold] = REGIME_BEAR
    regime[slope.isna()] = np.nan  # warmup bars excluded
    return regime


def classify_vol_regimes(
    prices: pd.Series,
    vol_window: int = VOL_WINDOW,
    high_q: float = VOL_HIGH_QUANTILE,
    low_q: float = VOL_LOW_QUANTILE,
) -> pd.Series:
    """Classify each bar as high_vol / low_vol / normal_vol by trailing realised vol.

    Realised vol = annualised std of daily returns over a trailing window.
    The high/low thresholds are the cross-sample quantiles of the vol series
    itself (computed once over the whole history). This is a *descriptive*
    segmentation, not a predictive one: we're labelling historical periods by
    their realised volatility, which is fine for stress-testing (we know the
    regime in hindsight). For live regime detection you'd use a rolling
    quantile; here the goal is to slice history, not predict.

    Args:
        prices:    Date-indexed price.
        vol_window: Trailing window for realised vol (default 63).
        high_q:    Quantile above which a bar is "high_vol".
        low_q:     Quantile below which a bar is "low_vol".

    Returns:
        Date-indexed Series of "high_vol"/"low_vol"/"normal_vol".
    """
    px = prices.astype(float)
    rets = px.pct_change()
    vol = rets.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(252)
    hi = vol.quantile(high_q)
    lo = vol.quantile(low_q)
    regime = pd.Series("normal_vol", index=px.index, dtype=object)
    regime[vol >= hi] = "high_vol"
    regime[vol <= lo] = "low_vol"
    regime[vol.isna()] = np.nan
    return regime


def _contiguous_slices(regime_labels: pd.Series, prices: pd.Series, min_bars: int = 63) -> list[RegimeSlice]:
    """Split a labelled series into contiguous runs of the same regime.

    A "slice" is a maximal run of bars sharing one label AND with no large date
    gap between them. We split on BOTH label changes AND date gaps (a gap > 5
    trading days means the bars aren't truly contiguous - concatenating them
    would create a false price jump). This is the fix for the bug where
    non-contiguous same-regime bars (e.g. two bear periods years apart) were
    merged into one slice, manufacturing fake multi-year "returns" at the gap.

    Slices shorter than ``min_bars`` are dropped - a 5-bar "bull" regime isn't
    a meaningful test period.
    """
    labels = regime_labels.dropna()
    if len(labels) == 0:
        return []
    # Boundary = label changes OR a date gap > 5 days (non-contiguous).
    label_change = labels.ne(labels.shift())
    date_gap = labels.index.to_series().diff() > pd.Timedelta(days=5)
    boundary = (label_change | date_gap).cumsum()

    slices: list[RegimeSlice] = []
    for _, group in labels.groupby(boundary):
        if len(group) < min_bars:
            continue
        regime = str(group.iloc[0])
        start, end = group.index[0], group.index[-1]
        seg_px = prices.loc[start:end]
        n = len(seg_px)
        ann_ret = (float(seg_px.iloc[-1]) / float(seg_px.iloc[0])) ** (252 / max(n, 1)) - 1 if n > 1 else 0.0
        slices.append(RegimeSlice(regime=regime, start=start, end=end,
                                  n_bars=n, ann_return=ann_ret))
    return slices


def build_regime_slices(
    prices: pd.Series,
    regime_types: tuple[str, ...] = ("trend", "vol"),
    min_bars: int = 63,
) -> dict[str, list[RegimeSlice]]:
    """Build all regime slices for the requested regime types.

    Args:
        prices:      Date-indexed price.
        regime_types: Which regime families to build ("trend" and/or "vol").
        min_bars:    Minimum slice length to keep.

    Returns:
        {regime_name: [RegimeSlice, ...]}. Each regime may have several slices
        (markets cycle through the same regime multiple times).
    """
    out: dict[str, list[RegimeSlice]] = {}
    if "trend" in regime_types:
        labels = classify_trend_regimes(prices)
        for reg in (REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS):
            slices = _contiguous_slices(labels.where(labels == reg), prices, min_bars)
            if slices:
                out[reg] = slices
    if "vol" in regime_types:
        labels = classify_vol_regimes(prices)
        for reg in ("high_vol", "low_vol", "normal_vol"):
            slices = _contiguous_slices(labels.where(labels == reg), prices, min_bars)
            if slices:
                out[reg] = slices
    return out


# ---------------------------------------------------------------------------
# Stress test runner
# ---------------------------------------------------------------------------

class StressTester:
    """Run a strategy comparison in each market regime.

    For each regime, concatenates that regime's slices into one price/pred
    series and re-runs the :class:`StrategyComparator`. The result per regime
    is a full ComparisonResult, so you can compare strategies WITHIN a regime
    (e.g. "who wins in bear markets?") and ACROSS regimes (e.g. "does the
    winner stay robust?").

    Args:
        comparator: A StrategyComparator (or factory config) for per-regime runs.
        min_bars:   Minimum slice length to consider (default 63 ~ 3 months).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        cost_model: Any | None = None,
        risk_config: dict | None = None,
        min_bars: int = 63,
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_model = cost_model
        self.risk_config = risk_config or {}
        self.min_bars = min_bars

    def run(
        self,
        strategies: list[Strategy],
        predictions: pd.DataFrame,
        prices: pd.Series,
        regime_types: tuple[str, ...] = ("trend",),
    ) -> StressTestResult:
        """Run the comparison in each regime and assemble cross-regime results.

        Each CONTIGUOUS regime slice is backtested independently (a slice is a
        maximal run of bars sharing one regime label). Concatenating
        non-contiguous slices would create false price jumps at the gaps (e.g. a
        bear slice ending 2009 and the next starting 2011 would show a fake
        multi-year "return" in one bar), corrupting the equity curve. So we run
        each slice as its own self-contained backtest and aggregate.

        Per regime, the strategy's return is the CAPITAL-WEIGHTED average of its
        slice returns (weighted by slice length, since each slice starts fresh
        at initial_capital). This is the honest cross-slice summary: a strategy
        that returns +20% in a 100-bar slice and -10% in a 200-bar slice
        averages to (+20*100 + -10*200)/300 = 0% - reflecting that it spent
        twice as long in the losing slice.

        Args:
            strategies:  Candidate strategies.
            predictions: Date-indexed model predictions (aligned to prices).
            prices:      Date-indexed price (Adj Close) - the regime source.
            regime_types: Which regime families to test.

        Returns:
            A StressTestResult with per-regime comparisons + a summary table.
        """
        px = prices.astype(float)
        preds = predictions.reindex(px.index) if predictions is not None else pd.DataFrame(index=px.index)
        slices_by_regime = build_regime_slices(px, regime_types, self.min_bars)

        result = StressTestResult()
        result.strategy_names = [getattr(s, "name", str(s)) for s in strategies]

        for regime, slices in slices_by_regime.items():
            # Backtest each contiguous slice independently, then aggregate.
            # Per-strategy: collect (return, n_bars) across slices for weighting.
            slice_comparisons: list[ComparisonResult] = []
            per_strategy_slices: dict[str, list[tuple[float, int]]] = {
                getattr(s, "name", str(s)): [] for s in strategies
            }
            total_bars = 0
            for sl in slices:
                reg_px = px.loc[sl.start:sl.end]
                reg_preds = preds.loc[sl.start:sl.end] if preds is not None else pd.DataFrame(index=reg_px.index)
                if len(reg_px) < self.min_bars:
                    continue
                comp = StrategyComparator(
                    initial_capital=self.initial_capital, cost_model=self.cost_model,
                    risk_config=self.risk_config,
                )
                comparison = comp.run(strategies, reg_preds, reg_px.to_frame())
                slice_comparisons.append(comparison)
                total_bars += len(reg_px)
                for row in comparison.rows:
                    per_strategy_slices[row.strategy_name].append((row.total_return, len(reg_px)))

            if not slice_comparisons or total_bars == 0:
                logger.info("regime %s: skipped (no usable slices)", regime)
                continue

            # Aggregate: build a synthetic ComparisonResult whose rows carry the
            # length-weighted average return per strategy. We reuse the LAST
            # slice's comparison as the structural template (metrics like sharpe
            # are from the longest slice, as a representative), but overwrite
            # total_return with the weighted average for the summary table.
            # The summary_table() reads total_return, so this is what matters.
            longest = max(slice_comparisons, key=lambda c: c.rows[0].n_trades if c.rows else 0) if slice_comparisons else None
            agg_rows = []
            if longest is not None:
                import copy
                for row in longest.rows:
                    slice_rets = per_strategy_slices.get(row.strategy_name, [])
                    if slice_rets:
                        tot_n = sum(n for _, n in slice_rets)
                        wavg = sum(r * n for r, n in slice_rets) / tot_n if tot_n > 0 else 0.0
                    else:
                        wavg = row.total_return
                    new_row = copy.copy(row)
                    new_row.total_return = wavg
                    new_row.n_trades = sum(
                        r.n_trades for c in slice_comparisons for r in c.rows
                        if r.strategy_name == row.strategy_name
                    )
                    agg_rows.append(new_row)
                # Re-rank by the aggregate score (reuse the comparator's scorer).
                comp_obj = StrategyComparator(
                    initial_capital=self.initial_capital, cost_model=self.cost_model,
                    risk_config=self.risk_config,
                )
                comp_obj._score(agg_rows)
                agg_rows.sort(key=lambda r: r.score, reverse=True)
                benchmark_ret = agg_rows[0].beats_benchmark  # placeholder
                comparison = ComparisonResult(
                    rows=agg_rows, benchmark_return=0.0,
                    recommended=agg_rows[0].strategy_name if agg_rows else "",
                    recommended_beats_benchmark=False,
                    results=[],
                )
            else:
                comparison = slice_comparisons[0]

            result.by_regime[regime] = RegimeResult(
                regime=regime, slices=slices, comparison=comparison,
                n_bars=total_bars,
            )
            top = comparison.rows[0] if comparison.rows else None
            logger.info(
                "regime %s: %d slices, %d total bars, top='%s' wavg_ret=%.2f%%",
                regime, len(slices), total_bars,
                top.strategy_name if top else "?",
                (top.total_return * 100) if top else 0.0,
            )
        return result


def robustness_verdict(stress: StressTestResult, benchmark_name: str = "buy_and_hold") -> dict[str, Any]:
    """Summarise cross-regime robustness into a verdict.

    A strategy is ROBUST if it beats the benchmark (or loses least) CONSISTENTLY
    across regimes - not just on average. This function computes, per strategy:
        - how many regimes it won (top scorer)
        - whether its return sign is stable (positive in every regime, or
          negative in every regime -> stable; flipping -> fragile)
        - its worst-regime return (the "how bad does it get" number)

    Returns:
        Dict with per-strategy robustness stats + an overall verdict string.
    """
    if not stress.by_regime:
        return {"verdict": "no regimes tested"}
    regimes = sorted(stress.by_regime.keys())
    out: dict[str, Any] = {"regimes": regimes, "strategies": {}}
    for name in stress.strategy_names:
        rets = []
        wins = 0
        for reg in regimes:
            rr = stress.by_regime[reg]
            match = [r for r in rr.comparison.rows if r.strategy_name == name]
            if match:
                rets.append(match[0].total_return)
                if rr.comparison.rows[0].strategy_name == name:
                    wins += 1
        if not rets:
            continue
        signs = {"+" if r > 0 else "-" for r in rets}
        stable = len(signs) == 1  # same sign everywhere
        out["strategies"][name] = {
            "regimes_won": wins,
            "regimes_tested": len(regimes),
            "worst_return": min(rets),
            "best_return": max(rets),
            "sign_stable": stable,
            "returns": dict(zip(regimes, rets)),
        }
    return out
