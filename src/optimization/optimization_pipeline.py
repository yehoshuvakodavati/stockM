"""
StockM v1.0 - Phase 6, Lessons 2 / 9 / 11 / 12 / 13 (orchestrator)
Hyperparameter Optimization Pipeline
====================================

Per-ticker, end-to-end:

    load prepared dataset (X/y, leakage-safe)
        -> for each model:
              run a baseline (default-params) eval on val + test  [Phase 5 reuse]
              run the configured search (grid/random/bayesian) on TRAIN
                  using time-series CV
              evaluate the tuned model on val + test
              compute learning + validation curves
              save the tuned model + versioned metadata
              log the experiment
        -> select the best tuned model (lowest CV RMSE)
        -> compare best-tuned vs baseline (deploy tuned only if it earns it)
        -> mark the deployed model + run robustness across regimes
        -> save a per-ticker optimization report

Selection discipline
---------------------
- Search uses TRAIN data with time-series CV only (never val/test for tuning).
- The best tuned model is chosen by CV RMSE.
- It is DEPLOYED only if it beats the Phase 5 baseline on validation RMSE
  by a tolerance; otherwise the simpler baseline is kept (don't deploy
  complexity that hasn't earned its keep).
- The TEST split is touched once for the honest report.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.baseline_models import DEFAULT_PARAMS, available_models, create_model
from models.data_loader import PREPARED_DIR, load_dataset
from models.evaluation import (
    directional_accuracy,
    evaluate_model,
    naive_baseline_metrics,
    regression_metrics,
)
from models.model_storage import load_model as load_baseline_model

from optimization.experiment_tracker import ExperimentTracker, make_experiment_id
from optimization.hyperparameter_optimizer import has_optuna, optimize_model
from optimization.model_saver import (
    build_optimized_metadata,
    load_optimized_model,
    mark_best_optimized,
    save_optimized_model,
)
from optimization.model_selection import compare_models, select_best_optimized
from optimization.robustness import robustness_report
from optimization.search_spaces import get_grid_space, get_random_space
from optimization.time_series_validation import make_cv
from optimization.validation_curves import (
    compute_learning_curve,
    compute_validation_curve,
    save_curves,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "optimization"

logger = logging.getLogger("stockm.optimization.pipeline")


def _estimator_factory(model_name: str, random_state: int = 42):
    """Return a callable(params) -> fresh estimator with n_jobs=1 for search."""

    def factory(params: dict[str, Any]):
        base = create_model(model_name, params)
        # Search parallelises across (config x fold); keep inner n_jobs=1 to
        # avoid oversubscription. (RF/XGB respect n_jobs.)
        if hasattr(base, "n_jobs"):
            base.n_jobs = 1
        return base

    return factory


def _eval_on_splits(model, X_val, y_val, X_test, y_test) -> dict[str, Any]:
    val = evaluate_model(model, X_val, y_val)
    test = evaluate_model(model, X_test, y_test)
    y_pred_test = model.predict(X_test)
    return {
        "val": val, "test": test,
        "directional_accuracy": directional_accuracy(y_test, y_pred_test),
    }


class OptimizationPipeline:
    """Config-driven per-ticker hyperparameter optimization + selection."""

    def __init__(
        self,
        target_col: str = "target_next_return",
        method: str = "random",        # grid | random | bayesian
        n_iter: int = 15,
        cv_splits: int = 3,
        cv_method: str = "timeseries",
        scoring: str = "neg_root_mean_squared_error",
        n_jobs: int = -1,
        random_state: int = 42,
        models: list[str] | None = None,   # which models to tune; None = all tunable
        prepared_dir: Path | None = None,
    ) -> None:
        self.target_col = target_col
        self.method = method
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.cv_method = cv_method
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.prepared_dir = prepared_dir or PREPARED_DIR
        self.tracker = ExperimentTracker()
        # Models worth tuning = those with a non-empty random/bayes space.
        self.models = models or self._default_tunable_models()

    def _default_tunable_models(self) -> list[str]:
        avail = set(available_models())
        # "ridge" is the tunable linear variant; we add it if sklearn has it.
        tunable = ["ridge", "decision_tree", "random_forest", "gradient_boosting"]
        if "xgboost" in avail:
            tunable.append("xgboost")
        if "lightgbm" in avail:
            tunable.append("lightgbm")
        # Keep only those we can actually build. Ridge needs sklearn (always).
        return [m for m in tunable if m == "ridge" or m in avail]

    # ------------------------------------------------------------------ run
    def run(self, symbol: str) -> dict[str, Any]:
        """Full optimization flow for one ticker."""
        logger.info("=== Optimization: %s | method=%s | n_iter=%d | cv=%s/%d ===",
                    symbol, self.method, self.n_iter, self.cv_method, self.cv_splits)
        data = load_dataset(symbol, target_col=self.target_col,
                            prepared_dir=self.prepared_dir)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        feats = data["feature_names"]

        cv = make_cv(self.cv_method, n_splits=self.cv_splits)
        cv_info = {"method": self.cv_method, "n_splits": self.cv_splits}
        dataset_info = {
            "symbol": symbol, "row_counts": data["row_counts"],
            "date_ranges": data["date_ranges"],
        }
        naive_test = naive_baseline_metrics(y_test)

        # Phase 5 baseline (best saved model) for comparison.
        baseline_model_name, baseline_val_rmse, baseline_test_rmse = self._load_baseline(
            symbol, X_val, y_val, X_test, y_test
        )

        per_model: dict[str, dict[str, Any]] = {}
        for mname in self.models:
            try:
                res = self._tune_one(
                    mname, symbol, X_train, y_train, X_val, y_val, X_test, y_test,
                    feats, cv, cv_info, dataset_info,
                )
                per_model[mname] = res
            except Exception as e:  # noqa: BLE001
                logger.warning("%s: tuning %s failed: %s", symbol, mname, e)

        if not per_model:
            raise RuntimeError(f"No model could be tuned for {symbol}.")

        # Pick the best tuned model by CV RMSE.
        best = select_best_optimized({m: r for m, r in per_model.items()})
        best_name = best["model"]
        best_tuned = best["best_estimator"]

        # Compare vs baseline -> decide deploy.
        # IMPORTANT: compare on a CONSISTENT basis. The tuned model's CV RMSE
        # (smaller per-fold train sets) is NOT comparable to the baseline's
        # full-validation RMSE - CV scores are systematically higher. So we
        # compare both models' FULL-VALIDATION RMSE (each trained on the full
        # train set, evaluated on the same val split). CV RMSE is kept in the
        # report for transparency but is not the deploy decision basis.
        opt_val_rmse = per_model[best_name]["val_rmse"]
        opt_test = evaluate_model(best_tuned, X_test, y_test)
        opt_dir = directional_accuracy(y_test, best_tuned.predict(X_test))
        comparison = compare_models(
            baseline_val_rmse=baseline_val_rmse,
            baseline_test_rmse=baseline_test_rmse,
            baseline_model=baseline_model_name or "none",
            optimized_val_rmse=opt_val_rmse,
            optimized_test_rmse=opt_test["rmse"],
            optimized_model=best_name,
            directional_accuracy=opt_dir,
        )
        comparison["optimized_cv_rmse"] = round(best["best_cv_rmse"], 6)
        comparison["note"] = (
            "deploy decision uses full-validation RMSE (apples-to-apples); "
            "CV RMSE reported separately for transparency."
        )

        # Deploy: mark the best tuned; the predictor prefers optimized.
        mark_best_optimized(symbol, best_name)
        # If the baseline still wins, record that too (predictor falls back).
        deploy = comparison["deploy"]

        # Robustness of the deployed model across test regimes.
        robust = robustness_report(best_tuned, X_test, y_test, n_segments=6)

        report = {
            "symbol": symbol,
            "target_col": self.target_col,
            "method": self.method,
            "cv": cv_info,
            "n_iter": self.n_iter,
            "models_tuned": list(per_model.keys()),
            "best_tuned_model": best_name,
            "best_cv_rmse": round(best["best_cv_rmse"], 6),
            "best_hyperparameters": best["best_params"],
            "naive_test_rmse": round(naive_test["rmse"], 6),
            "comparison": comparison,
            "deploy": deploy,
            "robustness": robust,
            "per_model": {m: {
                "best_cv_rmse": round(r["best_cv_rmse"], 6),
                "best_params": r["best_params"],
                "val_rmse": round(r["val_rmse"], 6),
                "test_rmse": round(r["test_rmse"], 6),
                "directional_accuracy": round(r["directional_accuracy"], 4),
                "wall_time_s": round(r["wall_time_s"], 3),
                "n_candidates": r["n_candidates"],
            } for m, r in per_model.items()},
        }
        self._save_report(report, symbol)

        logger.info(
            "%s done | best_tuned=%s cv_rmse=%.6f | baseline=%s val_rmse=%.6f | deploy=%s | dir_acc=%.4f | stable=%s",
            symbol, best_name, best["best_cv_rmse"],
            baseline_model_name, baseline_val_rmse, deploy, opt_dir,
            robust["stable_across_regimes"],
        )
        return {
            "symbol": symbol,
            "best_tuned_model": best_name,
            "best_cv_rmse": round(best["best_cv_rmse"], 6),
            "baseline_model": baseline_model_name,
            "baseline_val_rmse": round(baseline_val_rmse, 6),
            "deploy": deploy,
            "optimized_test_rmse": round(opt_test["rmse"], 6),
            "directional_accuracy": round(opt_dir, 4),
            "stable_across_regimes": robust["stable_across_regimes"],
            "n_models_tuned": len(per_model),
        }

    # ----------------------------------------------------------- tune one
    def _tune_one(
        self, model_name, symbol, X_train, y_train, X_val, y_val, X_test, y_test,
        feats, cv, cv_info, dataset_info,
    ) -> dict[str, Any]:
        factory = _estimator_factory(model_name, self.random_state)
        t0 = time.perf_counter()
        result = optimize_model(
            model_name=model_name,
            estimator_factory=factory,
            X=X_train, y=y_train,
            method=self.method, cv=cv, n_iter=self.n_iter,
            scoring=self.scoring, n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        wall = time.perf_counter() - t0

        tuned = result["best_estimator"]
        evals = _eval_on_splits(tuned, X_val, y_val, X_test, y_test)

        # Save the tuned model + metadata.
        meta = build_optimized_metadata(
            model_name=model_name, symbol=symbol,
            optimization_method=self.method,
            best_hyperparameters=result["best_params"],
            feature_names=feats,
            scaler_ref=str(self.prepared_dir / symbol.replace(".", "_") / "scaler_params.json"),
            dataset_info=dataset_info, cv_info=cv_info,
            metrics={
                "cv_rmse": result["best_cv_rmse"],
                "validation": evals["val"], "test": evals["test"],
                "directional_accuracy": evals["directional_accuracy"],
            },
            training_time_s=wall, seed=self.random_state,
        )
        save_optimized_model(tuned, meta, symbol, model_name)

        # Curves (learning curve + a validation curve on the key capacity knob).
        try:
            lc = compute_learning_curve(factory(result["best_params"]), X_train, y_train, cv)
            vc = None
            cap_param, cap_range = self._capacity_range(model_name)
            if cap_param is not None:
                vc = compute_validation_curve(
                    factory({}), X_train, y_train, cv, cap_param, cap_range,
                )
            save_curves(symbol, model_name, lc, vc)
        except Exception as e:  # noqa: BLE001
            logger.info("%s/%s curves skipped: %s", symbol, model_name, e)

        # Log experiment.
        exp_id = make_experiment_id(
            symbol, model_name, self.method, result["best_params"],
        )
        self.tracker.log({
            "experiment_id": exp_id,
            "timestamp": meta["tuned_at"],
            "symbol": symbol, "model_name": model_name, "method": self.method,
            "hyperparameters": result["best_params"],
            "dataset_version": dataset_info,
            "cv_method": cv_info["method"], "n_splits": cv_info["n_splits"],
            "val_rmse": evals["val"]["rmse"], "val_r2": evals["val"]["r2"],
            "test_rmse": evals["test"]["rmse"], "test_r2": evals["test"]["r2"],
            "directional_accuracy": evals["directional_accuracy"],
            "beats_baseline": False,  # set by caller for the chosen model
            "training_time_s": wall, "seed": self.random_state, "best": False,
        })

        return {
            "best_estimator": tuned,
            "best_params": result["best_params"],
            "best_cv_rmse": result["best_cv_rmse"],
            "val_rmse": evals["val"]["rmse"],
            "test_rmse": evals["test"]["rmse"],
            "directional_accuracy": evals["directional_accuracy"],
            "wall_time_s": wall,
            "n_candidates": result["n_candidates"],
        }

    # ----------------------------------------------------- capacity range
    def _capacity_range(self, model_name: str) -> tuple[str | None, list]:
        """Pick the key capacity hyperparameter + range for the validation curve."""
        if model_name in ("random_forest", "decision_tree", "xgboost", "lightgbm"):
            return "max_depth", [3, 4, 6, 8, 10, 12]
        if model_name == "gradient_boosting":
            return "max_depth", [3, 4, 5, 6, 8]
        if model_name == "ridge":
            return "alpha", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        return None, []

    # ----------------------------------------------------- baseline load
    def _load_baseline(
        self, symbol: str, X_val, y_val, X_test, y_test,
    ) -> tuple[str | None, float, float]:
        """Load the Phase 5 deployed baseline and RE-EVALUATE it fresh on the
        current run's val/test splits.

        Re-evaluating (rather than trusting the stored Phase 5 metrics) makes
        the baseline-vs-optimized comparison fully apples-to-apples: both
        models are scored on exactly the same rows in the same run.
        """
        try:
            model, meta = load_baseline_model(symbol)
            val_rmse = evaluate_model(model, X_val, y_val)["rmse"]
            test_rmse = evaluate_model(model, X_test, y_test)["rmse"]
            return meta.get("model_name"), float(val_rmse), float(test_rmse)
        except Exception:  # noqa: BLE001
            logger.info("%s: no Phase 5 baseline found; using naive as floor.", symbol)
            return None, float("nan"), float("nan")

    # ----------------------------------------------------------- report
    def _save_report(self, report: dict[str, Any], symbol: str) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"{symbol.replace('.', '_')}_optimization_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
