"""
StockM v1.0 - Phase 5, Lesson 8
Model Evaluation
================

Turns predictions into numbers you can reason about. Every metric here answers
a different question about how wrong the model is, and each has a distinct
financial reading.

Metrics
-------
MAE  - Mean Absolute Error: average |y - y_hat|. Same units as the target
       (return). Robust to outliers. "On average, how far off in return?"
MSE  - Mean Squared Error: average (y - y_hat)^2. Penalises large errors
       quadratically - so a few bad crash-day predictions dominate it.
RMSE - sqrt(MSE). Back to return units, but still outlier-sensitive. The
       standard "headline" regression metric.
R^2  - Coefficient of determination. Fraction of variance explained.
       1.0 = perfect, 0.0 = no better than predicting the mean, <0 = worse
       than the mean. In finance, R^2 on daily returns is typically TINY
       (0.005 - 0.05) - markets are mostly noise. A small positive R^2 is a
       real signal; do not expect 0.9.
MAPE - Mean Absolute Percentage Error. Relative error. Unstable when y is
       near zero (returns often are), so treat with care - included for
       completeness, not as the primary metric.

Naive baseline
--------------
We also compute the metrics for a model that predicts ZERO every day (the
drift-less naive baseline). If a trained model can't beat "always predict 0",
it has learned nothing useful. This contextualisation is the difference
between "RMSE = 0.018" (meaningless alone) and "RMSE = 0.018 vs naive 0.019
- barely better" (honest).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, guarded against near-zero y.

    Returns NaN if too few non-trivial points (so it doesn't mislead).
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    # Only count points where |y| is large enough to make a ratio meaningful.
    mask = np.abs(y_true) > 1e-4
    if mask.sum() < 10:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute the full regression metric suite.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dict with mae, mse, rmse, r2, mape.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = _mape(y_true, y_pred)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


def naive_baseline_metrics(y_true) -> dict[str, float]:
    """Metrics for the "always predict zero" drift-less naive model.

    Predicting zero return is the natural floor for a return-prediction
    model: it assumes no edge. Every trained model must beat this on RMSE/R^2
    to justify its existence.
    """
    y_true = np.asarray(y_true, dtype="float64")
    zero_pred = np.zeros_like(y_true)
    return regression_metrics(y_true, zero_pred)


def evaluate_model(model, X, y) -> dict[str, float]:
    """Predict with a fitted model and score it. Returns the metric dict."""
    y_pred = model.predict(X)
    return regression_metrics(y, y_pred)


def directional_accuracy(y_true, y_pred) -> float:
    """% of rows where the model gets the SIGN of the return right.

    Even a weak return regressor can be useful if its *direction* (UP/DOWN)
    is right more often than not - that is what a trading signal cares about.
    50% = coin flip; >52% consistently is a real edge in equities.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    # Only score rows with a non-trivial true move (ignore ~flat days).
    mask = np.abs(y_true) > 1e-4
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))
