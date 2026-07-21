"""
StockM v1.0 - Phase 5, Lessons 2 / 4 / 5 / 6 / 7
Baseline Model Factory
======================

One place that knows how to build every baseline estimator. Adding a new
model = one entry here; the training pipeline and comparison code never
change. That is the Open/Closed Principle applied to models.

Models (all REGRESSORS - the roadmap predicts next-day return)
--------------------------------------------------------------
LinearRegression (Lesson 4)
    y = w . x + b. Assumes a linear relationship, normally-distributed
    residuals, and (for stable coefficients) low feature correlation. Fast,
    interpretable, but high-bias: it cannot capture interactions or
    non-linearities. A strong *floor* baseline - if a tree model can't beat
    it, the signal is weak.

DecisionTreeRegressor (Lesson 5)
    Recursively splits the feature space to minimise variance in each leaf.
    Captures non-linearity and interactions for free, but a single deep tree
    is high-variance: it memorises the training set (overfit). We limit depth
    to control this.

RandomForestRegressor (Lesson 6)
    Ensemble of many de-correlated trees (bootstrap sampling + random feature
    subsets + averaging). Reduces variance vs a single tree without raising
    bias - the classic bagging win.

GradientBoosting (Lesson 7)
    Sequential ensemble: each tree fits the *residuals* of the previous one,
    correcting errors step by step (learning rate controls the step size).
    Often the best tabular model. Primary = sklearn HistGradientBoostingRegressor
    (dependency-free); XGBoost and LightGBM are auto-included if installed.

Why model training is isolated from feature engineering
-------------------------------------------------------
Feature engineering produces data; model training consumes it. Mixing them
couples two independent rates of change (features evolve, models evolve) and
makes leakage harder to enforce. This module owns ONLY estimator construction.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Protocol

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger("stockm.models")

# Optional boosters - auto-detected at import. If absent, they are simply not
# offered; HistGradientBoosting covers the boosting slot dependency-free.
try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor

    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False


class Estimator(Protocol):
    """Structural type for any sklearn-like regressor we build."""

    def fit(self, X, y): ...
    def predict(self, X): ...
    def get_params(self, deep: bool = True) -> dict: ...


# ---------------------------------------------------------------------------
# Default hyperparameters. Deliberately modest - these are *baselines*, not
# tuned production models. Tuning happens in a later (optuna) phase.
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "linear_regression": {},
    "decision_tree": {"max_depth": 6, "min_samples_leaf": 50, "random_state": 42},
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "n_jobs": -1,
        "random_state": 42,
    },
    "gradient_boosting": {  # sklearn HistGradientBoosting
        "max_iter": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "l2_regularization": 0.0,
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": -1,
    },
}


def _build_linear_regression(params: dict[str, Any]) -> Estimator:
    return LinearRegression(**params)


def _build_decision_tree(params: dict[str, Any]) -> Estimator:
    return DecisionTreeRegressor(**params)


def _build_random_forest(params: dict[str, Any]) -> Estimator:
    return RandomForestRegressor(**params)


def _build_gradient_boosting(params: dict[str, Any]) -> Estimator:
    return HistGradientBoostingRegressor(**params)


def _build_xgboost(params: dict[str, Any]) -> Estimator:
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed; install it or use gradient_boosting.")
    return XGBRegressor(**params)


def _build_lightgbm(params: dict[str, Any]) -> Estimator:
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed (optional).")
    return LGBMRegressor(**params)


_BUILDERS = {
    "linear_regression": _build_linear_regression,
    "decision_tree": _build_decision_tree,
    "random_forest": _build_random_forest,
    "gradient_boosting": _build_gradient_boosting,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
}


def available_models() -> list[str]:
    """Return the model names usable in this environment.

    XGBoost / LightGBM appear only if their libraries are importable.
    """
    names = ["linear_regression", "decision_tree", "random_forest", "gradient_boosting"]
    if _HAS_XGB:
        names.append("xgboost")
    if _HAS_LGBM:
        names.append("lightgbm")
    return names


def _filter_params(builder, params: dict[str, Any]) -> dict[str, Any]:
    """Drop params the estimator's constructor doesn't accept.

    Different boosting families use different names (XGBoost ``n_estimators``
    vs sklearn HistGBT ``max_iter``). Sharing a config block across families
    would otherwise crash the one that doesn't recognise a key. Filtering by
    signature keeps each family's overrides isolated and the pipeline robust
    to config drift.
    """
    try:
        sig = inspect.signature(builder)
        accepted = set(sig.parameters)
    except (ValueError, TypeError):  # pragma: no cover
        return dict(params)
    # ``**kwargs``-style constructors accept anything.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params)
    return {k: v for k, v in params.items() if k in accepted}


def create_model(name: str, params: dict[str, Any] | None = None) -> Estimator:
    """Instantiate a baseline estimator by name.

    Args:
        name:   One of available_models().
        params: Override the DEFAULT_PARAMS for this model.

    Returns:
        An unfitted sklearn-like regressor.
    """
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}. Available: {available_models()}")
    builder = _BUILDERS[name]
    merged = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}
    return builder(_filter_params(builder, merged))


def get_baseline_models(
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Estimator]:
    """Return {name: unfitted_estimator} for every available baseline.

    Args:
        overrides: Optional per-model hyperparameter overrides, keyed by name.
    """
    overrides = overrides or {}
    models: dict[str, Estimator] = {}
    for name in available_models():
        try:
            models[name] = create_model(name, overrides.get(name))
        except ImportError as e:
            logger.warning("skipping %s: %s", name, e)
    if not _HAS_XGB:
        logger.info("xgboost not installed - using gradient_boosting (HistGBT) as the booster.")
    return models
