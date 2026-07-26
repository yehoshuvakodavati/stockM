"""
StockM v1.0 - Phase 6, Lessons 4 / 5 / 6
Hyperparameter Optimizer
========================

Three search strategies behind one interface (``optimize_model``):

Grid Search (Lesson 4)
    Exhaustive over a discrete space. Cost = product of all list sizes, so
    it explodes with dimensions - fine for 2-3 hyperparameters, intractable
    beyond. Finds the best within the grid by definition, but the grid is a
    tiny slice of the real space.

Random Search (Lesson 5)
    Samples n_iter configs from distributions. Bergstra & Bengio (2012)
    showed random often finds 95%-as-good configs in 1/10 of grid's cost
    because most problems have a few *important* dimensions and many
    unimportant ones - random tries many values on the important ones
    instead of exhaustively on the unimportant ones.

Bayesian Optimization (Lesson 6)
    Builds a surrogate model (optuna: TPE by default) of the objective and
    uses an acquisition function to choose the next config to try based on
    all previous trials. Each trial informs the next, so it reaches a good
    config in fewer trials than random - but trials are sequential (hard
    to parallelise) and there is per-trial overhead. Best for expensive
    objectives where each fit is slow.

All three use a TIME-SERIES CV (never K-Fold) and a leak-free scoring
(neg_root_mean_squared_error by default). The test set is never used here.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from optimization.search_spaces import (
    get_bayes_space,
    get_grid_space,
    get_random_space,
)

logger = logging.getLogger("stockm.optimization")

try:
    import optuna  # type: ignore

    _HAVE_OPTUNA = True
except Exception:  # pragma: no cover
    _HAVE_OPTUNA = False

# Default scoring: RMSE (lower better) encoded as sklearn's negative score.
DEFAULT_SCORING = "neg_root_mean_squared_error"


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def grid_search(
    estimator,
    X,
    y,
    cv,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Exhaustive grid search over the model's GRID_SPACES entry."""
    space = get_grid_space(model_name) if model_name else {}
    if not space:
        raise ValueError(f"No grid space for {model_name!r}; use random/bayes.")
    search = GridSearchCV(
        estimator, space, cv=cv, scoring=scoring, n_jobs=n_jobs,
        refit=True, error_score="raise",
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    return _sklearn_result(search, "grid", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------
def random_search(
    estimator,
    X,
    y,
    cv,
    n_iter: int = 15,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
    random_state: int = 42,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Random search sampling n_iter configs from RANDOM_SPACES."""
    space = get_random_space(model_name) if model_name else {}
    if not space:
        raise ValueError(f"No random space for {model_name!r}.")
    search = RandomizedSearchCV(
        estimator, space, n_iter=n_iter, cv=cv, scoring=scoring,
        n_jobs=n_jobs, random_state=random_state, refit=True, error_score="raise",
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    return _sklearn_result(search, "random", time.perf_counter() - t0)


def _sklearn_result(search, method: str, wall: float) -> dict[str, Any]:
    """Normalise a fitted GridSearchCV/RandomizedSearchCV into a result dict."""
    best = search.best_estimator_
    return {
        "method": method,
        "best_estimator": best,
        "best_params": _jsonable_params(search.best_params_),
        "best_cv_score": float(search.best_score_),  # neg RMSE (higher=better)
        "best_cv_rmse": float(-search.best_score_),  # +RMSE (lower=better)
        "n_candidates": int(len(search.cv_results_["params"])),
        "wall_time_s": round(wall, 4),
        "cv_results": _summarise_cv_results(search.cv_results_),
    }


def _summarise_cv_results(cv_results: dict) -> list[dict[str, Any]]:
    """Compact the verbose cv_results_ into a list of per-config summaries."""
    rows: list[dict[str, Any]] = []
    means = cv_results["mean_test_score"]
    stds = cv_results["std_test_score"]
    times = cv_results.get("mean_fit_time", [0.0] * len(means))
    for i, params in enumerate(cv_results["params"]):
        rows.append({
            "params": _jsonable_params(params),
            "mean_score": float(means[i]),
            "std_score": float(stds[i]),
            "mean_fit_time": float(times[i]),
        })
    # Sort by score descending (best first).
    rows.sort(key=lambda r: r["mean_score"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Bayesian search (optuna)
# ---------------------------------------------------------------------------
def bayesian_search(
    estimator_factory: Callable[[dict], Any],
    X,
    y,
    cv,
    n_trials: int = 25,
    scoring: str = DEFAULT_SCORING,
    random_state: int = 42,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Bayesian (TPE) search using optuna. Sequential; trial-informs-trial.

    ``estimator_factory`` builds a fresh unfitted estimator from a params
    dict, so each trial starts clean (no state leakage between trials).
    """
    if not _HAVE_OPTUNA:
        raise ImportError(
            "optuna not installed; install it (`pip install optuna`) or use "
            "random_search instead."
        )
    suggest_fn = get_bayes_space(model_name) if model_name else None
    if suggest_fn is None:
        raise ValueError(f"No bayes space for {model_name!r}.")

    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)  # higher-is-better scorer (neg RMSE)
    splits = list(cv.split(X))

    def objective(trial) -> float:
        params = suggest_fn(trial)
        # Fit a fresh estimator per fold; average the CV score.
        scores: list[float] = []
        for tr_idx, va_idx in splits:
            est = estimator_factory(params)
            est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            scores.append(float(scorer(est, X.iloc[va_idx], y.iloc[va_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",  # maximize the (neg RMSE) scorer
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    wall = time.perf_counter() - t0

    best_params = _jsonable_params(study.best_params)
    # Refit a final estimator on the full CV-train data with the best params.
    best_estimator = estimator_factory(study.best_params)
    best_estimator.fit(X, y)

    return {
        "method": "bayesian",
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_cv_score": float(study.best_value),
        "best_cv_rmse": float(-study.best_value),
        "n_candidates": int(n_trials),
        "wall_time_s": round(wall, 4),
        "cv_results": [
            {"trial": t.number, "params": _jsonable_params(t.params),
             "mean_score": float(t.value) if t.value is not None else None}
            for t in study.trials
            if t.value is not None
        ],
    }


# ---------------------------------------------------------------------------
# Unified entry
# ---------------------------------------------------------------------------
def optimize_model(
    model_name: str,
    estimator_factory: Callable[[dict], Any],
    X,
    y,
    method: str = "random",
    cv=None,
    n_iter: int = 15,
    scoring: str = DEFAULT_SCORING,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Dispatch to grid / random / bayesian search by ``method``.

    Args:
        model_name:        Key into the search spaces.
        estimator_factory: callable(params) -> fresh unfitted estimator.
        X, y:              Training features + target.
        method:            "grid" | "random" | "bayesian".
        cv:                A time-series CV splitter (from time_series_validation).
        n_iter:            Trials for random / bayesian.
        scoring:           sklearn scoring string (default neg RMSE).
        n_jobs:            Parallelism for grid/random (inner estimator uses 1).
        random_state:      Seed.

    Returns:
        Normalised result dict (best_estimator, best_params, best_cv_rmse, ...).
    """
    # Build a base estimator with single-threaded inner n_jobs so the search
    # can parallelise across (config x fold) without oversubscribing.
    base = estimator_factory({})

    method = method.lower()
    if method == "grid":
        return grid_search(base, X, y, cv, scoring=scoring, n_jobs=n_jobs,
                           model_name=model_name)
    if method == "random":
        return random_search(base, X, y, cv, n_iter=n_iter, scoring=scoring,
                              n_jobs=n_jobs, random_state=random_state,
                              model_name=model_name)
    if method == "bayesian":
        return bayesian_search(estimator_factory, X, y, cv, n_trials=n_iter,
                               scoring=scoring, random_state=random_state,
                               model_name=model_name)
    raise ValueError(f"Unknown method {method!r}; use grid|random|bayesian.")


def _jsonable_params(params: dict) -> dict[str, Any]:
    """Coerce numpy/scipy values in a params dict to plain JSON types."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def has_optuna() -> bool:
    """Whether Bayesian optimization is available in this environment."""
    return _HAVE_OPTUNA
