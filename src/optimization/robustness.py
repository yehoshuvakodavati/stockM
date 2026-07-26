"""
StockM v1.0 - Phase 6, Lesson 10
Robustness Testing
=================

A single test-set number hides how the model behaves across regimes. A
model can have acceptable average RMSE yet collapse in bear markets - the
exact regime where a trading system loses money. Robustness testing slices
the test period into chronological segments and scores each, classifying
each segment as bull / bear / sideways from its realised trend.

Why segment by realised trend
-----------------------------
- Bull: cumulative return clearly positive - does the model add value in
  uptrends or just ride drift?
- Bear: cumulative return clearly negative - the stress test. Direction
  accuracy here is what survives a crash.
- Sideways: near-flat - choppy, mean-reverting; many false breakouts.

A model is "robust" only if its directional accuracy stays above ~50%
across ALL regimes, not just on average. Asymmetric regime performance is
a deployment red flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from models.evaluation import directional_accuracy, regression_metrics

# Trend thresholds for classifying a segment (cumulative return over it).
BULL_THRESHOLD = 0.05    # +5% cumulative => bull
BEAR_THRESHOLD = -0.05   # -5% cumulative => bear; between => sideways


def _classify_segment(cum_return: float) -> str:
    if cum_return >= BULL_THRESHOLD:
        return "bull"
    if cum_return <= BEAR_THRESHOLD:
        return "bear"
    return "sideways"


def robustness_report(
    model, X_test: pd.DataFrame, y_test: pd.Series, n_segments: int = 6,
) -> dict[str, Any]:
    """Score the model across chronological test segments + regime classes.

    Args:
        model:      Fitted model (Phase 6 optimized or Phase 5 baseline).
        X_test:     Test features (DatetimeIndex, sorted).
        y_test:     Test target.
        n_segments: Number of chronological segments to split the test set into.

    Returns:
        Dict with per-segment metrics, per-regime aggregates, and a
        ``stable`` flag (True iff directional accuracy > 0.5 in every regime
        that has enough rows to measure).
    """
    if not isinstance(X_test.index, pd.DatetimeIndex):
        raise TypeError("robustness_report requires a DatetimeIndex.")

    y_pred = np.asarray(model.predict(X_test), dtype="float64")
    y_true = np.asarray(y_test, dtype="float64")

    n = len(X_test)
    step = max(1, n // n_segments)
    bounds = list(range(0, n, step))
    if bounds[-1] != n:
        bounds.append(n)

    segments: list[dict[str, Any]] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 5:
            continue
        seg_true = y_true[a:b]
        seg_pred = y_pred[a:b]
        # Realised cumulative return of the segment (sum of log returns).
        cum = float(np.sum(seg_true))
        segments.append({
            "segment": i + 1,
            "date_range": (
                str(X_test.index[a].date()), str(X_test.index[b - 1].date())
            ),
            "n_rows": int(b - a),
            "regime": _classify_segment(cum),
            "cum_return": round(cum, 4),
            "rmse": round(float(np.sqrt(np.mean((seg_true - seg_pred) ** 2))), 6),
            "r2": round(float(1 - np.sum((seg_true - seg_pred) ** 2) /
                               max(np.sum((seg_true - seg_true.mean()) ** 2), 1e-12)), 4),
            "directional_accuracy": round(directional_accuracy(seg_true, seg_pred), 4),
        })

    # Aggregate by regime.
    regime_agg: dict[str, dict[str, Any]] = {}
    for s in segments:
        r = s["regime"]
        agg = regime_agg.setdefault(r, {"n_rows": 0, "dir_acc_sum": 0.0, "count": 0})
        agg["n_rows"] += s["n_rows"]
        agg["dir_acc_sum"] += s["directional_accuracy"]
        agg["count"] += 1
    for r, agg in regime_agg.items():
        agg["mean_directional_accuracy"] = round(agg["dir_acc_sum"] / agg["count"], 4)
        del agg["dir_acc_sum"]

    # Stable iff every regime's mean directional accuracy > 0.5.
    stable = all(
        agg["mean_directional_accuracy"] > 0.5
        for agg in regime_agg.values() if agg["count"] > 0
    )

    return {
        "n_segments": int(len(segments)),
        "segments": segments,
        "regime_summary": regime_agg,
        "stable_across_regimes": bool(stable),
    }
