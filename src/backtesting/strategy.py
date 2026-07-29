"""
StockM v1.0 - Phase 8, Lesson 3
Trading Strategy & Signal Generation
=====================================

Single Responsibility: decide WHAT to do - convert model predictions into
ordered trading Signals. This module deliberately separates the *decision*
from the *prediction* (models/prediction.py) and from the *execution*
(backtesting/portfolio.py):

    prediction (a raw number)  ->  Signal (an action)  ->  Trade (a fill)

Why signals are separated from prediction logic
------------------------------------------------
The prediction layer answers "what return do I expect?" (a continuous number).
Trading needs "what do I do?" (a discrete action + size + confidence). Coupling
them makes it impossible to reuse one model under different decision rules
(threshold vs confidence vs confirmation) without retraining. Keeping the rule
here means a new trading rule is a one-line config change, not a model change.

Two layers live here (both behind one responsibility - "decide the action"):
    SignalGenerator  - a stateless rule: prediction -> Signal   (Lesson 3)
    Strategy (ABC)   - a policy; may combine signals, price confirmation,
                       holding rules, etc. (Lesson 4). New strategies are added
                       by subclassing, never by editing the engine.

Three decision rules (Lesson 3), combinable:
    threshold:       act only if |predicted_return| > threshold (noise filter).
    confidence:      act only if a confidence proxy >= min_confidence. Our
                     models are REGRESSORS with no native probability, so
                     confidence is derived as a monotone function of
                     |predicted_return| (bigger move -> more confident). This is
                     the honest, model-agnostic proxy; a classifier would supply
                     a real probability instead.
    expected_return:  act only if the predicted return clears the round-trip
                     cost hurdle (predicted_return - cost_estimate has the sign
                     of the trade). This is where transaction costs first enter
                     the logic.

Dependency: imports only backtesting.trade (the vocabulary).
Implemented in: Lesson 3 (SignalGenerator), Lesson 4 (concrete strategies).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from backtesting.trade import Signal, TradeAction

logger = logging.getLogger("stockm.backtesting.strategy")

# Valid decision rules. A SignalGenerator may apply several at once (AND logic):
# a signal must pass EVERY active filter to become a trade.
VALID_RULES = frozenset({"threshold", "confidence", "expected_return"})


class SignalGenerator:
    """Turn raw model predictions into discrete Signals via decision rule(s).

    A generator is **stateless**: the same predictions in produce the same
    signals out, so changing a rule is a pure config edit with no hidden state.
    Multiple rules combine with AND - a prediction must clear *every* active
    filter to become a BUY/SELL; otherwise it is HOLD.

    Args:
        rules:         Subset of {"threshold", "confidence", "expected_return"}.
                       A rule is active only if named here.
        threshold:     Band for the threshold rule: act iff |r_hat| > threshold.
        min_confidence:Floor for the confidence rule (in [0,1)); None disables
                       it even if "confidence" is in rules (overridable below).
        cost_estimate: Round-trip cost fraction for the expected-return rule
                       (e.g. 0.002 = 20 bps). BUY requires r_hat - cost > 0;
                       SELL requires r_hat - cost < 0. NB: a SELL only clears the
                       cost hurdle when r_hat < -cost, i.e. the model predicts a
                       DOWN move larger than the cost - otherwise HOLD.
        confidence_scale: tau in conf = 1 - exp(-|r_hat|/tau). Defaults to a
                       typical daily-return std (~0.013 for NSE large caps) so
                       confidence is calibrated to realistic magnitudes.
        symbol:        Ticker tag written onto every Signal (the predictions
                       table may not carry it; the engine usually knows it).

    Design note on confidence for regressors
    ---------------------------------------
    A regressor outputs an expected return, not a probability. We need a
    confidence in [0,1] for the confidence rule, so we derive one from the only
    magnitude information we have - the size of the predicted move:

        conf(r_hat) = 1 - exp(-|r_hat| / tau)

    This is monotone in |r_hat|, bounded in [0,1), and 0 at r_hat=0. ``tau``
    sets the scale: with tau ~ 0.013 (daily-return std), a prediction of one
    std (~1.3%) gives conf ~ 0.63, and a 2-sigma move gives ~0.86. A real
    classifier would replace this with its calibrated P(UP); the interface
    (a ``confidence`` column) stays the same.
    """

    # Rules that determine DIRECTION (BUY vs SELL). ``confidence`` is NOT
    # directional: it is a magnitude filter, so it cannot act alone - combining
    # it with at least one directional rule is required (enforced in __init__).
    _DIRECTIONAL_RULES = frozenset({"threshold", "expected_return"})

    def __init__(
        self,
        rules: str | list[str] = ("threshold",),
        threshold: float = 0.0,
        min_confidence: float | None = None,
        cost_estimate: float = 0.0,
        confidence_scale: float = 0.013,
        symbol: str = "",
    ) -> None:
        # Normalise ``rules`` to a set of active filter names.
        if isinstance(rules, str):
            rules = [rules]
        active = set(rules)
        bad = active - VALID_RULES
        if bad:
            raise ValueError(
                f"unknown rule(s) {sorted(bad)}; valid: {sorted(VALID_RULES)}"
            )
        # A confidence-only generator cannot decide BUY vs SELL - confidence is
        # a magnitude filter with no direction. Require a directional rule too.
        if active and not (active & self._DIRECTIONAL_RULES):
            raise ValueError(
                "confidence is not directional: combine it with at least one "
                "of threshold / expected_return (rules that set BUY vs SELL)."
            )
        self.rules = active
        self.threshold = float(threshold)
        self.min_confidence = min_confidence
        self.cost_estimate = float(cost_estimate)
        self.confidence_scale = float(confidence_scale)
        self.symbol = symbol
        if self.confidence_scale <= 0:
            raise ValueError("confidence_scale must be > 0")
        logger.debug(
            "SignalGenerator rules=%s threshold=%s min_conf=%s cost=%s tau=%s",
            sorted(self.rules), self.threshold, self.min_confidence,
            self.cost_estimate, self.confidence_scale,
        )

    # -- confidence proxy --------------------------------------------------
    def confidence(self, r_hat: float) -> float:
        """Confidence in [0,1) derived from |predicted_return|.

        conf = 1 - exp(-|r_hat| / tau). Monotone, 0 at r_hat=0, saturates to 1.
        A classifier would override this with a calibrated P(UP); the
        ``confidence`` field on the Signal keeps the same contract.
        """
        return float(1.0 - np.exp(-abs(r_hat) / self.confidence_scale))

    # -- single-row decision ------------------------------------------------
    def _decide(self, r_hat: float) -> tuple[TradeAction, float]:
        """Apply the active rule(s) to one prediction; return (action, conf).

        Rules combine with AND: every active filter must agree on the direction
        (and clear its hurdle) for a BUY/SELL; any disagreement -> HOLD. This
        prevents, e.g., a threshold-passing but cost-failing prediction from
        becoming a money-losing trade.
        """
        # Direction from the threshold / expected-return rules.
        # threshold rule: BUY if r_hat > +t, SELL if r_hat < -t.
        # expected_return rule: BUY if r_hat - cost > 0, SELL if r_hat - cost < 0.
        vote_buy = True
        vote_sell = True

        if "threshold" in self.rules:
            vote_buy &= r_hat > self.threshold
            vote_sell &= r_hat < -self.threshold

        if "expected_return" in self.rules:
            net = r_hat - self.cost_estimate
            vote_buy &= net > 0.0
            # SELL clears the cost hurdle only when the predicted DOWN move
            # exceeds the cost (r_hat < -cost), else the round-trip loses money.
            vote_sell &= r_hat + self.cost_estimate < 0.0

        conf = self.confidence(r_hat)

        if "confidence" in self.rules and self.min_confidence is not None:
            passes = conf >= self.min_confidence
            vote_buy &= passes
            vote_sell &= passes

        if vote_buy:
            return TradeAction.BUY, conf
        if vote_sell:
            return TradeAction.SELL, conf
        return TradeAction.HOLD, conf

    # -- batch decision -----------------------------------------------------
    def generate(self, predictions: pd.DataFrame) -> list[Signal]:
        """Map each prediction row to a BUY / SELL / HOLD Signal.

        Args:
            predictions: Frame with a ``predicted_return`` column. The index
                          is used as the signal date (DatetimeIndex expected).
                          An optional ``confidence`` column, if present, is
                          preferred over the derived proxy (lets a classifier
                          supply a real probability). An optional ``symbol``
                          column overrides the generator's ``symbol``.

        Returns:
            Ordered list of Signals (one per row), including HOLDs so the
            timeline stays complete - the portfolio/engine skip HOLDs.

        Raises:
            KeyError if ``predicted_return`` is absent.
        """
        if "predicted_return" not in predictions.columns:
            raise KeyError(
                "predictions must have a 'predicted_return' column; got "
                f"{list(predictions.columns)}"
            )

        sym_col = "symbol" in predictions.columns
        conf_col = "confidence" in predictions.columns
        signals: list[Signal] = []

        for ts, row in predictions.iterrows():
            r_hat = float(row["predicted_return"])
            action, derived_conf = self._decide(r_hat)
            # Prefer a supplied confidence (classifier) over the derived proxy.
            conf = float(row["confidence"]) if conf_col else derived_conf
            sym = str(row["symbol"]) if sym_col and pd.notna(row.get("symbol")) else self.symbol
            # reason names the active rules for the audit trail / report.
            reason = "rule:" + "+".join(sorted(self.rules)) if self.rules else "rule:none"
            signals.append(
                Signal(
                    date=ts.date() if hasattr(ts, "date") else ts,
                    symbol=sym,
                    action=action,
                    predicted_return=r_hat,
                    confidence=conf,
                    strength=conf,  # confidence doubles as sizing strength (L5)
                    reason=reason,
                )
            )
        logger.info(
            "generated %d signals (%d BUY, %d SELL, %d HOLD) from %d rows",
            len(signals),
            sum(s.action == TradeAction.BUY for s in signals),
            sum(s.action == TradeAction.SELL for s in signals),
            sum(s.action == TradeAction.HOLD for s in signals),
            len(predictions),
        )
        return signals


# ---------------------------------------------------------------------------
# Strategy interface + registry (Lesson 4)
# ---------------------------------------------------------------------------

# Registry of strategy classes by name, populated by @register_strategy at
# class-definition time. THIS IS THE EXTENSION POINT: adding a new strategy =
# write a new subclass + decorate it; nothing else in the codebase changes.
_STRATEGY_REGISTRY: dict[str, type["Strategy"]] = {}


def register_strategy(cls: type["Strategy"]) -> type["Strategy"]:
    """Class decorator: register a Strategy subclass under its ``name``.

    A new strategy in a new module just needs to be imported once (e.g. via the
    package __init__ or a runner) for its decorator to fire and make it
    discoverable through :func:`get_strategy`. The engine never hard-codes
    strategy names - it resolves them from the registry.
    """
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__}: a Strategy must define a class `name`")
    if name in _STRATEGY_REGISTRY and _STRATEGY_REGISTRY[name] is not cls:
        logger.warning(
            "strategy %r re-registered by %s (was %s)",
            name, cls.__name__, _STRATEGY_REGISTRY[name].__name__,
        )
    _STRATEGY_REGISTRY[name] = cls
    return cls


def available_strategies() -> list[str]:
    """Names of all registered strategies (sorted for stable output)."""
    return sorted(_STRATEGY_REGISTRY)


def get_strategy(name: str, **kwargs: Any) -> "Strategy":
    """Instantiate a registered strategy by name. The engine's factory hook."""
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(
            f"unknown strategy {name!r}; available: {available_strategies()}"
        )
    return _STRATEGY_REGISTRY[name](**kwargs)


def _date_of(ts: Any):
    """Normalise a pandas Timestamp / date-like to a datetime.date."""
    return ts.date() if hasattr(ts, "date") else ts


def _price_series(prices: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalise the price input to a single indexed Series (Adj Close).

    Accepts a Series (already one symbol) or a DataFrame with a single price
    column / an explicit ``adj_close`` column. v1 is single-symbol; the
    multi-symbol case is a portfolio-level concern (Lesson 5).
    """
    if isinstance(prices, pd.Series):
        return prices.astype(float)
    if isinstance(prices, pd.DataFrame):
        cols = [str(c).lower() for c in prices.columns]
        if "adj_close" in cols:
            return prices.iloc[:, cols.index("adj_close")].astype(float)
        if prices.shape[1] == 1:
            return prices.iloc[:, 0].astype(float)
    raise TypeError(
        "prices must be a Series or a single-column / adj_close DataFrame; "
        f"got {type(prices).__name__}"
    )


class Strategy(ABC):
    """Abstract base for all trading strategies.

    Open-Closed Principle: add strategies by subclassing + implementing
    :meth:`generate_signals` and decorating with :func:`register_strategy`;
    never modify the engine or existing strategies. The BacktestEngine speaks
    ONLY to this interface, so any registered subclass is automatically
    backtestable via ``get_strategy(name, **kwargs)``.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(
        self, predictions: pd.DataFrame, prices: pd.DataFrame
    ) -> list[Signal]:
        """Return the ordered list of Signals for the whole backtest window.

        Args:
            predictions: Date-indexed table with at least ``predicted_return``
                          (and optionally ``confidence``) per symbol. May be
                          empty for price-only strategies (e.g. Buy & Hold).
            prices:       Date-indexed canonical price series (Adj Close), as a
                          Series or single-column DataFrame.

        Note:
            Signals MUST respect look-ahead discipline: a signal at date t may
            use only prices with index <= t. Trailing rolling windows are
            look-ahead-safe; the engine re-checks ordering, but strategies are
            responsible for not peeking into the future.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete strategies (Lesson 4)
# ---------------------------------------------------------------------------

@register_strategy
class BuyAndHoldStrategy(Strategy):
    """Buy on the first bar, hold forever. The benchmark every strategy must beat.

    Uses no model predictions - it buys on day 1 and holds to the end. This is
    the honest baseline: if a prediction strategy can't beat Buy & Hold after
    costs, the model adds no value (our expected result given ~50% directional
    accuracy). Its dates come from ``prices`` (or ``predictions`` if prices is
    empty) so the timeline matches the backtest window exactly.
    """

    name = "buy_and_hold"

    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol

    def generate_signals(
        self, predictions: pd.DataFrame, prices: pd.DataFrame
    ) -> list[Signal]:
        # Prefer the prices index (the actual tradable dates); fall back to preds.
        idx = prices.index if prices is not None and len(prices) else predictions.index
        sym = self.symbol
        if not sym and predictions is not None and "symbol" in getattr(predictions, "columns", []):
            sym = str(predictions["symbol"].iloc[0])
        signals: list[Signal] = []
        for i, ts in enumerate(idx):
            action = TradeAction.BUY if i == 0 else TradeAction.HOLD
            signals.append(Signal(date=_date_of(ts), symbol=sym, action=action, reason="buy_and_hold"))
        logger.info("buy_and_hold: %d signals (1 BUY + %d HOLD)", len(signals), max(len(signals) - 1, 0))
        return signals


@register_strategy
class PredictionBasedStrategy(Strategy):
    """Trade in the direction the model predicts. The general prediction policy.

    Composes a :class:`SignalGenerator` (Lesson 3) so the decision rule is the
    single source of truth for "prediction -> action". By varying the
    generator's rules you get threshold / confidence / cost-aware variants
    without subclassing - but the named presets below (ThresholdStrategy,
    ConfidenceStrategy) exist for clarity in the Lesson 10 comparison.
    """

    name = "prediction_based"

    def __init__(
        self,
        rules: str | list[str] = ("threshold",),
        threshold: float = 0.0,
        min_confidence: float | None = None,
        cost_estimate: float = 0.0,
        confidence_scale: float = 0.013,
        symbol: str = "",
    ) -> None:
        self.generator = SignalGenerator(
            rules=rules, threshold=threshold, min_confidence=min_confidence,
            cost_estimate=cost_estimate, confidence_scale=confidence_scale, symbol=symbol,
        )

    def generate_signals(
        self, predictions: pd.DataFrame, prices: pd.DataFrame
    ) -> list[Signal]:
        return self.generator.generate(predictions)


@register_strategy
class ThresholdStrategy(PredictionBasedStrategy):
    """Act only when |predicted_return| exceeds a band (a noise filter).

    A preset of PredictionBasedStrategy with the threshold rule. ``threshold=0``
    reduces to pure direction (trade every day); larger bands trade less,
    saving transaction costs. The Lesson 3 exercise showed cost drag can exceed
    100% of capital when trading every day - this is the lever that controls it.
    """

    name = "threshold"

    def __init__(self, threshold: float = 0.002, symbol: str = "", **kwargs: Any) -> None:
        super().__init__(rules="threshold", threshold=threshold, symbol=symbol, **kwargs)


@register_strategy
class ConfidenceStrategy(PredictionBasedStrategy):
    """Act only on high-confidence predictions (magnitude-filtered).

    A preset using threshold (direction) + confidence (filter). Requires the
    model to be "sure enough" (confidence >= min_confidence) before trading.
    Since our confidence is derived from |predicted_return|, this trades only
    on larger expected moves - fewer trades, lower cost drag.
    """

    name = "confidence"

    def __init__(self, min_confidence: float = 0.5, threshold: float = 0.0, symbol: str = "", **kwargs: Any) -> None:
        super().__init__(rules=["threshold", "confidence"], threshold=threshold,
                        min_confidence=min_confidence, symbol=symbol, **kwargs)


@register_strategy
class MovingAverageConfirmationStrategy(Strategy):
    """Trade with the model only when the trend confirms it.

    Combines a prediction-driven direction (via a SignalGenerator) with a
    moving-average trend filter on the price:
        - BUY  only if model says UP   AND price > MA  (uptrend confirms)
        - SELL only if model says DOWN AND price < MA  (downtrend confirms)
        - HOLD otherwise (model and trend disagree, or MA not yet available)

    The MA is a TRAILING rolling mean (window ending at t), so it is
    look-ahead-safe. The first ``ma_window-1`` bars have no MA -> HOLD (no
    confirmation possible). This filters counter-trend signals - the most
    common reason a 50%-accurate model bleeds money is trading against the
    trend, and the MA filter vetoes exactly those.
    """

    name = "ma_confirmation"

    def __init__(
        self,
        ma_window: int = 20,
        threshold: float = 0.0,
        min_confidence: float | None = None,
        cost_estimate: float = 0.0,
        symbol: str = "",
    ) -> None:
        if ma_window < 2:
            raise ValueError("ma_window must be >= 2")
        self.ma_window = int(ma_window)
        # Direction comes from the threshold rule; confidence is an optional
        # extra filter on top of the MA confirmation.
        rules = ("threshold",) if min_confidence is None else ["threshold", "confidence"]
        self.generator = SignalGenerator(
            rules=rules, threshold=threshold, min_confidence=min_confidence,
            cost_estimate=cost_estimate, symbol=symbol,
        )

    def generate_signals(
        self, predictions: pd.DataFrame, prices: pd.DataFrame
    ) -> list[Signal]:
        base = self.generator.generate(predictions)  # direction from the model
        px = _price_series(prices)
        ma = px.rolling(self.ma_window, min_periods=self.ma_window).mean()
        # Align MA / price to the signal dates by position (robust to a date
        # in the predictions index that is missing from prices -> NaN -> HOLD).
        sig_dates = pd.DatetimeIndex([pd.Timestamp(s.date) for s in base])
        ma_a = ma.reindex(sig_dates)
        px_a = px.reindex(sig_dates)

        confirmed: list[Signal] = []
        for i, sig in enumerate(base):
            m_val, p_val = ma_a.iloc[i], px_a.iloc[i]
            if pd.isna(m_val) or pd.isna(p_val):
                confirmed.append(replace(sig, action=TradeAction.HOLD, reason=sig.reason + "+ma:na"))
                continue
            if sig.action == TradeAction.BUY and p_val > m_val:
                confirmed.append(replace(sig, reason=sig.reason + "+ma:up"))
            elif sig.action == TradeAction.SELL and p_val < m_val:
                confirmed.append(replace(sig, reason=sig.reason + "+ma:down"))
            else:
                confirmed.append(replace(sig, action=TradeAction.HOLD, reason=sig.reason + "+ma:veto"))
        logger.info("ma_confirmation(window=%d): %d signals", self.ma_window, len(confirmed))
        return confirmed
