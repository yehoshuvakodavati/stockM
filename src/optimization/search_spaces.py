"""
StockM v1.0 - Phase 6, Lesson 3
Hyperparameter Search Spaces
=============================

Defines, per model, which hyperparameters to search and over what ranges.
This is the single source of truth for the "knobs and ranges" of every
search method (grid / random / Bayesian) in this phase.

Why spaces are bounded, not open
--------------------------------
An unbounded search space overfits the *validation set itself*: with enough
configs, you find one that got lucky on validation but won't generalize.
We keep spaces principled, regularisation-leaning (finance punishes
variance), and modest in size so the search is tractable across 50 tickers.

Hyperparameter reference (the roadmap's "explain every hyperparameter")
-----------------------------------------------------------------------
max_depth (trees/ensembles)
    Max tree depth. Capacity lever: deeper = more regions = lower bias,
    higher variance + overfit risk. Finance default leans shallow (4-10).
min_samples_split
    Min rows required to split a node. Higher = more conservative (less
    memorisation). Regularising.
min_samples_leaf
    Min rows in a leaf. Higher = coarser leaves = lower variance, higher
    bias. Strong regulariser; very effective at preventing lone-point leaves.
n_estimators (RF / boosting)
    Number of trees. RF: more trees reduce variance (diminishing returns).
    Boosting: more trees fit harder (capacity) - pair with a LOWER
    learning_rate to avoid overfit; cap with early stopping.
learning_rate (boosting)
    Step size per tree (shrinkage). Lower = slower, more regularised
    learning, needs MORE trees. Lower LR + more trees usually generalises
    best, at higher compute cost.
subsample (boosting / RF rows)
    Row sampling fraction per tree. <1 = stochastic = regularising (each
    tree sees less data, less memorisation). Typical 0.6-0.9.
colsample_bytree (boosting)
    Feature sampling fraction per tree. <1 = decorrelates trees +
    regularises. 0.6-1.0.
max_features (RF / trees)
    Features considered per split. Lower = more decorrelated trees (lower
    variance) but higher bias. "sqrt" / "log2" / float are common.
regularization (reg_alpha / reg_lambda / l2_regularization / alpha)
    L1/L2 penalty on the learned function. Higher = lower variance,
    higher bias. alpha (Ridge/Lasso) and l2_regularization (HistGBT) are
    the linear/boosting equivalents of the same idea.

Search-space conventions
------------------------
- GRID spaces:   dict[param -> list]            (exhaustive over the lists)
- RANDOM spaces: dict[param -> scipy distribution] (sampled n_iter times)
- BAYES spaces:  a callable(trial) -> dict using optuna suggest_* (so the
  surrogate can model each hyperparameter's distribution shape).

The three are kept separate because grid needs discrete lists, random
needs scipy distributions, and Bayesian needs optuna suggest calls - they
cannot share one structure without losing clarity.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# GRID spaces (small, exhaustive). Keep tiny: grid cost = product of sizes.
# --------------------------------------------------------------------------
GRID_SPACES: dict[str, dict[str, list]] = {
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [6, 10, None],
        "min_samples_leaf": [10, 20, 50],
        "max_features": ["sqrt", 1.0],
    },
    "gradient_boosting": {  # sklearn HistGradientBoosting
        "max_iter": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_regularization": [0.0, 0.1],
    },
    "decision_tree": {
        "max_depth": [4, 6, 10, None],
        "min_samples_leaf": [20, 50, 100],
        "max_features": ["sqrt", 1.0],
    },
    # Linear regression has essentially no tunable knobs for OLS; we expose
    # Ridge (L2) as the tunable linear variant in the random/bayes spaces.
    "linear_regression": {},  # nothing to grid-search for plain OLS
    "xgboost": {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "lightgbm": {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, -1],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
}


# --------------------------------------------------------------------------
# RANDOM spaces (scipy distributions). n_iter samples drawn.
# --------------------------------------------------------------------------
def _random_spaces() -> dict[str, dict[str, Any]]:
    return {
        "ridge": {  # tunable linear: L2-regularised linear regression
            "alpha": stats.loguniform(1e-3, 1e2),
        },
        "decision_tree": {
            "max_depth": stats.randint(3, 12),
            "min_samples_leaf": stats.randint(10, 100),
            "max_features": ["sqrt", "log2", 1.0],  # discrete list is fine in RandomizedSearchCV
        },
        "random_forest": {
            "n_estimators": stats.randint(100, 300),
            "max_depth": stats.randint(4, 12),
            "min_samples_leaf": stats.randint(5, 50),
            "max_features": ["sqrt", "log2", 0.5, 1.0],
        },
        "gradient_boosting": {
            "max_iter": stats.randint(150, 400),
            "max_depth": stats.randint(3, 9),
            "learning_rate": stats.loguniform(1e-3, 3e-1),
            "l2_regularization": stats.loguniform(1e-4, 1e0),
        },
        "xgboost": {
            "n_estimators": stats.randint(150, 400),
            "max_depth": stats.randint(3, 9),
            "learning_rate": stats.loguniform(1e-3, 3e-1),
            "subsample": stats.uniform(0.6, 0.4),     # [0.6, 1.0]
            "colsample_bytree": stats.uniform(0.6, 0.4),
            "reg_lambda": stats.loguniform(1e-3, 1e1),
        },
        "lightgbm": {
            "n_estimators": stats.randint(150, 400),
            "num_leaves": stats.randint(15, 63),
            "learning_rate": stats.loguniform(1e-3, 3e-1),
            "subsample": stats.uniform(0.6, 0.4),
            "colsample_bytree": stats.uniform(0.6, 0.4),
            "reg_lambda": stats.loguniform(1e-3, 1e1),
        },
    }


RANDOM_SPACES = _random_spaces()


# --------------------------------------------------------------------------
# BAYESIAN spaces (optuna suggest calls). One callable per model.
# --------------------------------------------------------------------------
def _bayes_ridge(trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}


def _bayes_decision_tree(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
    }


def _bayes_random_forest(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
    }


def _bayes_gradient_boosting(trial) -> dict:
    return {
        "max_iter": trial.suggest_int("max_iter", 150, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1e0, log=True),
    }


def _bayes_xgboost(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e1, log=True),
    }


def _bayes_lightgbm(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e1, log=True),
    }


BAYES_SPACES: dict[str, Callable[[Any], dict]] = {
    "ridge": _bayes_ridge,
    "decision_tree": _bayes_decision_tree,
    "random_forest": _bayes_random_forest,
    "gradient_boosting": _bayes_gradient_boosting,
    "xgboost": _bayes_xgboost,
    "lightgbm": _bayes_lightgbm,
}


def get_grid_space(model_name: str) -> dict[str, list]:
    """Return the grid search space for a model (empty dict = nothing to grid)."""
    return GRID_SPACES.get(model_name, {})


def get_random_space(model_name: str) -> dict[str, Any]:
    """Return the random search space (scipy dists / lists) for a model."""
    return RANDOM_SPACES.get(model_name, {})


def get_bayes_space(model_name: str) -> Callable[[Any], dict] | None:
    """Return the optuna suggest callable for a model, or None if none."""
    return BAYES_SPACES.get(model_name)
