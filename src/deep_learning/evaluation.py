"""
StockM v1.0 - Phase 7, Lesson 9
Deep-Learning Evaluation + ML-vs-DL Comparison
================================================

The definitive "does DL beat the optimized ML models?" answer.

What this module does
---------------------
1. Full metric suite per DL model: MAE, MSE, RMSE, R^2, MAPE, directional
   accuracy, beats_naive - REUSING models.evaluation so DL is scored with the
   EXACT same yardsticks the classical baselines were (apples-to-apples).
2. Timing: training time (from the Trainer) + inference time (timed test pass).
3. Resource: device (CPU/GPU) + parameter count (capacity/memory proxy).
4. ML-vs-DL comparison on the SAME test dates: the DL models predict on
   windowed sequences (lose the first `lookback-1` test days); the ML model
   predicts on every test row. We align both to the DL models' overlapping
   dates so the comparison is on identical days, identical target, identical
   metrics - no cheating.

The honest bar (unchanged across phases)
---------------------------------------
A model "wins" only if it beats the naive zero-predictor on RMSE AND clears
~50% directional accuracy. Per [[dl-baseline-findings]], no architecture has
cleared this yet on RELIANCE daily returns. This lesson reports the honest
numbers, whatever they are.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Reuse the Phase-5 metric suite so DL and ML are scored identically.
from models.evaluation import (
    directional_accuracy,
    naive_baseline_metrics,
    regression_metrics,
)
from models.data_loader import load_split

logger = logging.getLogger("stockm.deep_learning.evaluation")


def compute_dl_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Full metric suite for a DL model's predictions.

    Reuses models.evaluation.regression_metrics + directional_accuracy so the
    numbers are directly comparable to the classical baseline reports.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    reg = regression_metrics(y_true, y_pred)
    naive = naive_baseline_metrics(y_true)
    return {
        **reg,                                    # mae, mse, rmse, r2, mape
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "naive_rmse": naive["rmse"],
        "beats_naive": reg["rmse"] < naive["rmse"],  # the honest floor
        "n_samples": int(len(y_true)),
    }


def timed_inference(
    model: "torch.nn.Module", test_dl, device: str, task: str = "regression"
) -> tuple[np.ndarray, float]:
    """Run a timed inference pass over the test set.

    Returns:
        (predictions, inference_time_seconds). Predictions are sigmoid(logits)
        for classification, raw values for regression.
    """
    model.eval()
    preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in test_dl:
            out = model(xb.to(device)).squeeze(-1).cpu()
            if task == "classification":
                out = torch.sigmoid(out)
            preds.append(out)
    inference_time = time.perf_counter() - t0
    y_pred = torch.cat(preds).numpy()
    return y_pred, inference_time


def evaluate_dl_model(
    model: "torch.nn.Module", test_dl, y_true: np.ndarray, task: str, device: str
) -> dict[str, Any]:
    """Full DL evaluation: metrics + inference time on the test set."""
    y_pred, inf_time = timed_inference(model, test_dl, device, task)
    metrics = compute_dl_metrics(y_true, y_pred)
    metrics["inference_time_s"] = round(inf_time, 4)
    return metrics


def evaluate_ml_on_dates(
    symbol: str, dates: pd.Index, target_col: str = "target_next_return"
) -> dict[str, Any]:
    """Evaluate the deployed ML model on the SAME dates the DL models predict.

    Loads the optimized (or baseline) ML model via models.prediction, predicts
    on the test split, and restricts to `dates` (the DL windows' target dates)
    so ML and DL are compared on identical days.

    Returns:
        Dict with model name, predictions (aligned), metrics, inference time.
    """
    from models.prediction import load_model  # lazy: avoid hard dep at import

    model, meta = load_model(symbol)
    model_name = meta.get("model_name", "unknown")
    df = load_split(symbol, "test")

    feature_names = meta.get("feature_names", [])
    X = df[feature_names] if feature_names else df.drop(
        columns=[c for c in df.columns if c.startswith("target_")]
    )
    y = df[target_col]

    # Restrict to the DL models' dates (the overlapping window).
    common = dates.intersection(X.index)
    X_al = X.loc[common]
    y_al = y.loc[common]

    t0 = time.perf_counter()
    y_pred = np.asarray(model.predict(X_al), dtype=np.float64)
    inf_time = time.perf_counter() - t0

    metrics = compute_dl_metrics(y_al.values, y_pred)
    metrics["inference_time_s"] = round(inf_time, 4)
    metrics["model_name"] = model_name
    metrics["source"] = meta.get("source", "unknown")
    metrics["n_samples"] = int(len(y_al))
    return metrics


# ---------------------------------------------------------------------------
# The definitive ML-vs-DL comparison.
# Run:  PYTHONPATH=src python src/deep_learning/evaluation.py RELIANCE.NS
# ---------------------------------------------------------------------------
def compare_ml_vs_dl(
    symbol: str, task: str = "regression", epochs: int = 30
) -> dict[str, Any]:
    """Train all 4 DL models, evaluate the deployed ML model, compare on the
    same test dates with the full metric suite + timing.

    Args:
        symbol: Ticker.
        task:   "regression" (target_next_return) or "classification".
        epochs: DL training epochs (early stopping may cut this short).
    """
    from deep_learning.framework import (
        get_device,
        make_dataloader,
        make_tensor_dataset,
        set_global_seed,
        load_training_config,
    )
    from deep_learning.sequence_builder import build_sequences
    from deep_learning.lstm import build_lstm
    from deep_learning.gru import build_gru
    from deep_learning.cnn import build_cnn
    from deep_learning.transformer import build_transformer
    from deep_learning.trainer import Trainer, TrainSettings

    tcfg = load_training_config()
    device = get_device(tcfg["device"])
    target_col = "target_next_return" if task == "regression" else "target_direction"
    seq = build_sequences(symbol, target_col=target_col)

    dtype_y = torch.float32 if task == "classification" else None
    train_ds = make_tensor_dataset(seq["X_train_seq"], seq["y_train_seq"], dtype_y=dtype_y)
    val_ds = make_tensor_dataset(seq["X_validation_seq"], seq["y_validation_seq"], dtype_y=dtype_y)
    test_ds = make_tensor_dataset(seq["X_test_seq"], seq["y_test_seq"], dtype_y=dtype_y)
    train_dl = make_dataloader(train_ds, tcfg["batch_size"], shuffle=True, drop_last=True)
    val_dl = make_dataloader(val_ds, tcfg["batch_size"], shuffle=False)
    test_dl = make_dataloader(test_ds, tcfg["batch_size"], shuffle=False)

    y_true = seq["y_test_seq"].astype(np.float64)
    dates_test = seq["dates_test"]

    common = dict(epochs=epochs, lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"],
                  grad_clip=tcfg["gradient_clip_norm"], early_stopping=True,
                  es_patience=10, es_min_delta=1e-4, save_best=False, checkpoint_dir=None,
                  log_every=10)
    n_feat = seq["n_features"]

    factories = {
        "lstm": (lambda: build_lstm(n_feat, task=task), 0),
        "gru": (lambda: build_gru(n_feat, task=task), 0),
        "cnn": (lambda: build_cnn(n_feat, task=task), 0),
        "transformer": (lambda: build_transformer(n_feat, task=task), 5),  # warmup
    }

    results: dict[str, Any] = {}
    for name, (factory, warmup) in factories.items():
        logger.info("=== %s ===", name.upper())
        set_global_seed(42)
        settings = TrainSettings(warmup_epochs=warmup, label=name.upper(), **common)
        trainer = Trainer(settings, device=device)
        train_res = trainer.train(factory().to(device), train_dl, val_dl, test_dl, task=task)
        # The Trainer restored best weights into trainer.last_model; evaluate the
        # FULL metric suite (R2/MAPE/inference_time) on that exact model.
        eval_res = evaluate_dl_model(trainer.last_model, test_dl, y_true, task, device)
        results[name] = {
            "params": train_res["params"],
            "train_time_s": train_res["train_time_s"],
            "epochs_run": train_res["epochs_run"],
            "best_val_loss": train_res["best_val_loss"],
            **eval_res,
        }

    # ML model on the same dates
    logger.info("=== ML (deployed) ===")
    results["ml_optimized"] = evaluate_ml_on_dates(symbol, dates_test, target_col)

    # Print the definitive table
    cols = ["params", "train_time_s", "inference_time_s", "rmse", "mae", "r2",
            "mape", "directional_accuracy", "beats_naive"]
    header = "  %-12s" % "metric" + "".join(f"  {n:>11}" for n in ["ML_opt", "LSTM", "GRU", "CNN", "TRANS"])
    logger.info("ML-vs-DL | %s %s | aligned on %d test dates", symbol, task, results["ml_optimized"]["n_samples"])
    logger.info(header)
    for c in cols:
        vals = [results[k].get(c) for k in ["ml_optimized", "lstm", "gru", "cnn", "transformer"]]
        row = "  %-12s" % c + "".join(
            f"  {v:>11.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool)
            else f"  {str(v):>11}" for v in vals
        )
        logger.info(row)

    return {"symbol": symbol, "task": task, "results": results}


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    compare_ml_vs_dl(sym, task="regression", epochs=30)
