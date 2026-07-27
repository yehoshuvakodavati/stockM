"""
StockM v1.0 - Phase 7, Lesson 10
Model Comparison Report
=======================

Assembles a professional comparison report across ALL StockM models
(classical ML baselines + optimized ML + the 4 deep-learning architectures),
recommends the best for StockM, and documents the accuracy / speed /
interpretability / computational-cost trade-offs.

Design: the report GENERATES from existing artifacts, it does not retrain.
- Classical ML baselines: read from reports/training/<SYM>_baseline_report.json
  (the per-ticker Phase-5 report - already evaluated).
- Optimized ML + DL: the verified Lesson-9 comparison results, embedded as
  constants (reproducible by re-running deep_learning.evaluation.compare_ml_vs_dl).
This is the production pattern: a report consumes prior results; training is
expensive and belongs in its own pipeline run.

Output: reports/deep_learning/comparison_report.md (+ a machine-readable .json).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("stockm.deep_learning.comparison")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_REPORTS = PROJECT_ROOT / "reports" / "training"
DL_REPORTS = PROJECT_ROOT / "reports" / "deep_learning"


# ---------------------------------------------------------------------------
# Verified Lesson-9 results for RELIANCE.NS (regression, aligned 735 test
# dates, Trainer with early stopping + warmup, full metric suite reusing
# models.evaluation). Reproduce via:
#   from deep_learning.evaluation import compare_ml_vs_dl
#   compare_ml_vs_dl("RELIANCE.NS", task="regression", epochs=30)
# Embedding keeps the report generator instant + self-contained; a --rerun
# flag could call compare_ml_vs_dl fresh for a live snapshot.
# ---------------------------------------------------------------------------
_VERIFIED_OPTIMIZED_ML = {
    "lightgbm_optimized": {
        "test_rmse": 0.0132, "test_r2": -0.006, "directional_accuracy": 0.4938,
        "beats_naive": False, "train_time_s": None, "inference_time_s": 0.029,
        "params": None, "family": "classical_optimized",
        "interpretability": "Medium (tree importance + SHAP)",
    },
}
_VERIFIED_DL_RESULTS = {
    "lstm": {
        "test_rmse": 0.0162, "test_r2": -0.517, "directional_accuracy": 0.4924,
        "beats_naive": False, "train_time_s": 15.14, "inference_time_s": 0.086,
        "params": 60481, "family": "deep_learning",
        "interpretability": "Low (opaque cell state)",
    },
    "gru": {
        "test_rmse": 0.0145, "test_r2": -0.215, "directional_accuracy": 0.4952,
        "beats_naive": False, "train_time_s": 63.51, "inference_time_s": 0.148,
        "params": 45377, "family": "deep_learning",
        "interpretability": "Low (opaque hidden state)",
    },
    "cnn": {
        "test_rmse": 0.0178, "test_r2": -0.827, "directional_accuracy": 0.5048,
        "beats_naive": False, "train_time_s": 6.86, "inference_time_s": 0.038,
        "params": 20161, "family": "deep_learning",
        "interpretability": "Low-Medium (filters somewhat readable)",
    },
    "transformer": {
        "test_rmse": 0.0438, "test_r2": -10.09, "directional_accuracy": 0.5048,
        "beats_naive": False, "train_time_s": 137.83, "inference_time_s": 0.080,
        "params": 102657, "family": "deep_learning",
        "interpretability": "Medium (attention weights introspectable)",
    },
}

# Interpretability ratings for the classical baselines (qualitative dimension).
_BASELINE_INTERPRETABILITY = {
    "linear_regression": "High (coefficients = factor loadings)",
    "decision_tree": "High (human-readable splits)",
    "random_forest": "Medium (feature importance, many trees)",
    "gradient_boosting": "Medium (feature importance)",
}


def load_ml_baseline_results(symbol: str) -> dict[str, dict[str, Any]]:
    """Read the per-ticker ML baseline report; return {model_name: metrics}.

    Extracts test RMSE/R2/MAE, directional accuracy, training time, and the
    naive floor from reports/training/<SYM>_baseline_report.json.
    """
    path = TRAINING_REPORTS / f"{symbol.replace('.', '_')}_baseline_report.json"
    if not path.exists():
        raise FileNotFoundError(f"baseline report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    naive = report.get("naive_baseline", {}).get("test", {})
    out: dict[str, dict[str, Any]] = {"naive_zero": {
        "test_rmse": naive.get("rmse"), "test_r2": naive.get("r2"),
        "directional_accuracy": None, "beats_naive": True,  # the floor itself
        "train_time_s": 0.0, "inference_time_s": 0.0, "params": 0,
        "family": "naive", "interpretability": "N/A (predicts zero)",
    }}
    for name, m in report.get("models", {}).items():
        test = m.get("test", {})
        out[name] = {
            "test_rmse": test.get("rmse"), "test_r2": test.get("r2"),
            "directional_accuracy": m.get("directional_accuracy"),
            "beats_naive": (test.get("rmse", 1e9) < naive.get("rmse", 1e9)) if naive else None,
            "train_time_s": m.get("training_time_s"), "inference_time_s": None,
            "params": None, "family": "classical_baseline",
            "interpretability": _BASELINE_INTERPRETABILITY.get(name, "Medium"),
        }
    return out


def build_comparison_table(symbol: str) -> dict[str, dict[str, Any]]:
    """Assemble the unified model-comparison table for a ticker."""
    table = load_ml_baseline_results(symbol)
    table.update(_VERIFIED_OPTIMIZED_ML)
    table.update(_VERIFIED_DL_RESULTS)
    return table


def recommend_best(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pick the best model by TEST RMSE, with the honest edge caveat.

    Selection rule: lowest test RMSE among non-naive models. We ALSO flag
    whether the winner actually beats naive (none do yet) - because "best of a
    no-edge field" is not the same as "deployable edge".
    """
    candidates = {k: v for k, v in table.items() if k != "naive_zero" and v.get("test_rmse") is not None}
    best_name = min(candidates, key=lambda k: candidates[k]["test_rmse"])
    best = candidates[best_name]
    any_beats_naive = any(v.get("beats_naive") for v in candidates.values())
    return {
        "best_model": best_name,
        "test_rmse": best["test_rmse"],
        "test_r2": best["test_r2"],
        "directional_accuracy": best["directional_accuracy"],
        "beats_naive": best["beats_naive"],
        "any_model_beats_naive": any_beats_naive,
        "verdict": (
            "edge exists - deployable" if best["beats_naive"]
            else "no edge - best of a no-edge field; not deployable for trading"
        ),
    }


def _trade_off_analysis(table: dict[str, dict[str, Any]]) -> list[str]:
    """Prose discussion of accuracy / speed / interpretability / cost trade-offs."""
    lines = [
        "## Trade-off analysis (accuracy vs speed vs interpretability vs cost)",
        "",
        "**Accuracy (test RMSE / R2):** Every model has NEGATIVE R2 - all are worse",
        "than predicting the mean. The optimized LightGBM is closest to the naive",
        "floor (R2 -0.006); the Transformer is catastrophically overfit (R2 -10).",
        "Negative R2 across the board means there is essentially no signal to model",
        "in daily next-day returns from 40 OHLCV features on one ticker.",
        "",
        "**Speed:** Classical ML trains in seconds and infers in ~30 ms. The CNN is",
        "the DL speed champion (7 s train, 38 ms infer). The Transformer is the",
        "slowest (138 s train) and the GRU is anomalously slow on this CPU build",
        "(63 s). For end-of-day trading across 50 tickers, ML or CNN inference cost",
        "is negligible; recurrent/Transformer cost is acceptable but not free.",
        "",
        "**Interpretability:** linear_regression and decision_tree are fully",
        "interpretable (coefficients / splits). Tree ensembles (RF, GBM, LightGBM)",
        "offer feature importance + SHAP. The CNN's filters are somewhat readable;",
        "the Transformer's attention weights are introspectable (a Lesson-13 lever);",
        "LSTM/GRU cell states are opaque. For a regulated/auditable trading system,",
        "the ML models win on explainability.",
        "",
        "**Computational cost:** params range 0 (naive) to 103k (Transformer). A",
        "rule of thumb: params >> training samples (3540) invites overfitting -",
        "exactly what the Transformer's R2 -10 shows. The CNN (20k params) is the",
        "best-parametrized DL model and correspondingly overfit least among DL.",
        "",
        "**Cost-benefit verdict:** The optimized ML model gives the best accuracy",
        "AND the best speed AND the best interpretability AND the lowest cost. DL",
        "loses on every axis for THIS task. DL's potential value is forward-looking",
        "(richer features, pooled tickers, multimodal) - not for beating ML on",
        "single-ticker OHLCV daily returns.",
    ]
    return lines


def generate_comparison_report(symbol: str) -> Path:
    """Generate + save the professional model-comparison report for a ticker."""
    table = build_comparison_table(symbol)
    rec = recommend_best(table)

    # Sort models by test RMSE ascending (best first); naive first for reference.
    ordered = ["naive_zero"] + sorted(
        [k for k in table if k != "naive_zero" and table[k].get("test_rmse") is not None],
        key=lambda k: table[k]["test_rmse"],
    )

    lines = [
        f"# StockM - Model Comparison Report: {symbol}",
        "",
        f"Target: `target_next_return` (next-day log return regression).",
        f"Evaluated on the held-out TEST split (DL aligned to 735 windowed test",
        f"dates; ML on the full 764 test rows). Same metrics as the Phase-5",
        f"baselines (`models.evaluation`) - apples-to-apples.",
        "",
        "## Unified comparison (sorted by test RMSE, best first)",
        "",
        f"| rank | model | family | test_rmse | test_r2 | dir_acc | beats_naive | train_s | infer_s | params | interpretability |",
        f"|------|-------|--------|-----------|---------|---------|-------------|---------|----------|--------|------------------|",
    ]
    for i, name in enumerate(ordered, 1):
        m = table[name]
        rmse = f"{m['test_rmse']:.4f}" if m["test_rmse"] is not None else "-"
        r2 = f"{m['test_r2']:.3f}" if m["test_r2"] is not None else "-"
        da = f"{m['directional_accuracy']:.4f}" if m["directional_accuracy"] is not None else "-"
        bn = "yes" if m.get("beats_naive") else "no"
        tr = f"{m['train_time_s']:.1f}" if m["train_time_s"] is not None else "-"
        inf = f"{m['inference_time_s']:.3f}" if m["inference_time_s"] is not None else "-"
        par = f"{m['params']:,}" if m["params"] else "-"
        lines.append(f"| {i} | {name} | {m['family']} | {rmse} | {r2} | {da} | {bn} | {tr} | {inf} | {par} | {m['interpretability']} |")
    lines.append("")

    lines += [
        "## Recommendation",
        "",
        f"**Best model by test RMSE: `{rec['best_model']}`** (RMSE {rec['test_rmse']:.4f}, "
        f"R2 {rec['test_r2']:.3f}, directional accuracy {rec['directional_accuracy']:.4f}).",
        "",
        f"**Verdict: {rec['verdict']}.** "
        f"{'The winner beats the naive zero-predictor.' if rec['beats_naive'] else 'The winner does NOT beat the naive zero-predictor - it is merely the least-bad of a no-edge field.'}",
        "",
        "**Honest caveat:** None of the models clears the two-part deployability bar",
        "(beats_naive=True AND directional accuracy clearly >50%). The optimized",
        "LightGBM is the best StockM currently has, but it has no real trading",
        "edge on this task. Deploying it to live/paper trading would, at best, lose",
        "money on transaction costs. The bottleneck is the SIGNAL (negative R2",
        "everywhere), not the architecture.",
        "",
    ]
    lines += _trade_off_analysis(table)
    lines += [
        "",
        "## Path to real edge (non-architecture levers)",
        "",
        "1. **Classification on `target_direction`** (UP/DOWN) instead of return",
        "   magnitude - the configured primary; matches the model_config.yaml",
        "   `output_size: 1 binary: UP probability` intent.",
        "2. **Longer horizon** (`target_return_5d`) - 5-day returns are less noisy",
        "   than daily.",
        "3. **Richer features** - enable the `technical`/`fundamental`/`macro`",
        "   groups (only `ohlcv` is active); more signal per row.",
        "4. **Pool tickers** - train one model across 50 NIFTY names for ~50x more",
        "   data; DL's capacity advantage emerges at scale.",
        "5. **Tune signal thresholds vs transaction costs** - even a 51% directional",
        "   model can be profitable if you only act on high-confidence calls.",
        "",
        "## Artifacts",
        "",
        "- ML baselines: `reports/training/<SYM>_baseline_report.json`",
        "- Optimized ML: `models/optimized/<SYM>/` (deployed via best_optimized.json)",
        "- DL checkpoints: `models/checkpoints/<SYM>_<model>.pt` (Lesson 8)",
        "- DL learning curves: `reports/deep_learning/<SYM>_<model>_curves.png`",
        "- This report: `reports/deep_learning/comparison_report.md` (+ .json)",
    ]

    DL_REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = DL_REPORTS / "comparison_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = DL_REPORTS / "comparison_report.json"
    json_path.write_text(json.dumps(
        {"symbol": symbol, "recommendation": rec, "table": table}, indent=2, default=str
    ), encoding="utf-8")
    logger.info("comparison report saved -> %s", md_path)
    logger.info("recommendation: best=%s | beats_naive=%s | verdict=%s",
                rec["best_model"], rec["beats_naive"], rec["verdict"])
    return md_path


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    generate_comparison_report(sym)
