"""
StockM v1.0 - Phase 5, Lessons 2 / 8 / 9 / 10 / 11 / 13
Baseline Training Orchestrator
==============================

End-to-end flow for one ticker:

    load prepared dataset (X/y, leakage-safe)
        -> for each baseline model:
              train on TRAIN, time it
              evaluate on VAL (selection) and TEST (honest report)
              extract feature importance
              residual / error analysis
              save model + versioned metadata
        -> build comparison, recommend best, mark it deployed
        -> save per-ticker report

Single Responsibility: this orchestrator owns sequencing and the
train/evaluate/save boundary. Estimator construction lives in
baseline_models, metrics in evaluation, persistence in model_storage.

Selection discipline
--------------------
Best model = lowest VALIDATION RMSE. Test is reported but never used for
selection (no future leakage in the deployment decision).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from models.baseline_models import get_baseline_models
from models.comparison import build_comparison, recommend_best
from models.data_loader import PREPARED_DIR, load_dataset
from models.error_analysis import residual_analysis
from models.evaluation import (
    directional_accuracy,
    evaluate_model,
    naive_baseline_metrics,
    regression_metrics,
)
from models.feature_importance import consensus_importance, rank_features
from models.model_storage import SAVED_MODELS_DIR, build_metadata, mark_best, save_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "training"

logger = logging.getLogger("stockm.models.training")


class BaselineTrainer:
    """Config-driven baseline training for one ticker at a time.

    Parameters
    ----------
    target_col : str
        Regression target column (default next-day log return).
    model_overrides : dict | None
        Per-model hyperparameter overrides, e.g. {"random_forest": {"n_estimators": 400}}.
    signal_threshold : float
        Threshold for the BUY/HOLD/SELL signal in error analysis context.
    """

    def __init__(
        self,
        target_col: str = "target_next_return",
        model_overrides: dict[str, dict[str, Any]] | None = None,
        signal_threshold: float = 0.0,
        prepared_dir: Path | None = None,
        saved_models_dir: Path | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self.target_col = target_col
        self.model_overrides = model_overrides or {}
        self.signal_threshold = signal_threshold
        self.prepared_dir = prepared_dir or PREPARED_DIR
        self.saved_models_dir = saved_models_dir or SAVED_MODELS_DIR
        self.reports_dir = reports_dir or REPORTS_DIR

    # ------------------------------------------------------------------ run
    def run(self, symbol: str) -> dict[str, Any]:
        """Train, evaluate, compare, save for one ticker. Returns a summary."""
        logger.info("=== Baseline training: %s (target=%s) ===", symbol, self.target_col)
        data = load_dataset(symbol, target_col=self.target_col, prepared_dir=self.prepared_dir)
        if data["dropped_na_train"]:
            logger.warning("%s: dropped %d NaN rows from train", symbol, data["dropped_na_train"])

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        feats = data["feature_names"]

        models = get_baseline_models(self.model_overrides)
        if not models:
            raise RuntimeError(f"No models available for {symbol}.")

        # Naive floor (predict zero) on val + test - everything must beat this.
        naive_val = naive_baseline_metrics(y_val)
        naive_test = naive_baseline_metrics(y_test)

        per_model: dict[str, dict[str, Any]] = {}
        importances: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, dict[str, Any]] = {}

        for name, estimator in models.items():
            logger.info("training %s / %s ...", symbol, name)
            t0 = time.perf_counter()
            estimator.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            val_metrics = evaluate_model(estimator, X_val, y_val)
            test_metrics = evaluate_model(estimator, X_test, y_test)

            # Directional accuracy on test (trading-relevant).
            y_pred_test = estimator.predict(X_test)
            dir_acc = directional_accuracy(y_test, y_pred_test)

            per_model[name] = {
                "training_time_s": train_time,
                "val": val_metrics,
                "test": test_metrics,
                "directional_accuracy": dir_acc,
            }
            importances[name] = rank_features(estimator, feats, top_n=len(feats))
            errors[name] = residual_analysis(
                y_test, y_pred_test, X=X_test, n_examples=5
            )

            # Persist this model + its provenance.
            meta = build_metadata(
                model_name=name,
                symbol=symbol,
                target_col=self.target_col,
                feature_names=feats,
                hyperparameters=estimator.get_params(),
                training_time_s=train_time,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                dataset_info={
                    "row_counts": data["row_counts"],
                    "date_ranges": data["date_ranges"],
                    "split_method": "chronological_gap1",
                },
                scaler_ref=str(self.prepared_dir / symbol.replace(".", "_") / "scaler_params.json"),
                is_best=False,  # set below after comparison
            )
            save_model(estimator, meta, symbol, name, self.saved_models_dir)

        # --- Compare + recommend + deploy ---------------------------------
        comparison = build_comparison(per_model, naive_test_rmse=naive_test["rmse"])
        recommendation = recommend_best(comparison)
        best_name = recommendation["best_model"]

        if best_name:
            mark_best(symbol, best_name, self.saved_models_dir)
            # Flip the is_best flag in the winner's metadata for traceability.
            self._mark_best_metadata(symbol, best_name)

        # Consensus feature importance across all models.
        fitted_models = {n: m for n, m in models.items()}
        consensus = consensus_importance(fitted_models, feats, top_n=len(feats))

        report = {
            "symbol": symbol,
            "target_col": self.target_col,
            "naive_baseline": {"val": naive_val, "test": naive_test},
            "models": per_model,
            "comparison": comparison,
            "recommendation": recommendation,
            "feature_importance": importances,
            "consensus_importance": consensus,
            "error_analysis": errors,
            "row_counts": data["row_counts"],
            "date_ranges": data["date_ranges"],
            "n_features": len(feats),
        }
        self._save_report(report, symbol)

        logger.info(
            "%s done | best=%s | val_rmse=%.6f test_rmse=%.6f dir_acc=%.4f naive_test_rmse=%.6f",
            symbol, best_name,
            per_model[best_name]["val"]["rmse"] if best_name else float("nan"),
            per_model[best_name]["test"]["rmse"] if best_name else float("nan"),
            per_model[best_name]["directional_accuracy"] if best_name else float("nan"),
            naive_test["rmse"],
        )
        return {
            "symbol": symbol,
            "best_model": best_name,
            "val_rmse": per_model[best_name]["val"]["rmse"] if best_name else None,
            "test_rmse": per_model[best_name]["test"]["rmse"] if best_name else None,
            "directional_accuracy": per_model[best_name]["directional_accuracy"] if best_name else None,
            "beats_naive": comparison[0]["beats_naive"] if comparison else False,
            "n_models": len(models),
        }

    # ------------------------------------------------------------- helpers
    def _mark_best_metadata(self, symbol: str, best_name: str) -> None:
        out_dir = self.saved_models_dir / symbol.replace(".", "_")
        meta_path = out_dir / f"{best_name}_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["is_best"] = True
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _save_report(self, report: dict[str, Any], symbol: str) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{symbol.replace('.', '_')}_baseline_report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
