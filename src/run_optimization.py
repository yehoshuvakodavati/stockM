"""
StockM v1.0 - Phase 6: Hyperparameter Optimization
Entry point for the optimization batch pipeline.

Flow (per ticker):
    load data/prepared/<SYMBOL>/{train,validation,test}.csv  (X/y, leak-safe)
        -> for each tunable model:
              search hyperparameters on TRAIN with time-series CV
                (grid / random / bayesian)
              evaluate tuned model on val + test
              save tuned model + versioned metadata -> models/optimized/<SYM>/
              log experiment -> experiments/optimization_runs.csv
              compute learning + validation curves -> reports/optimization/curves/
        -> select best tuned model (lowest CV RMSE)
        -> compare vs Phase 5 baseline; deploy tuned only if it earns it
        -> robustness across bull/bear/sideways test regimes
        -> per-ticker optimization report

Finally, aggregates a global leaderboard of optimized results.

Config is read from configs/optimization_config.yaml.
Run from the project root:
    python src/run_optimization.py                       # default subset
    python src/run_optimization.py RELIANCE.NS            # one ticker
    python src/run_optimization.py --all                  # every ticker
    python src/run_optimization.py RELIANCE.NS TCS.NS     # explicit subset
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization.hyperparameter_optimizer import has_optuna
from optimization.optimization_pipeline import (
    OptimizationPipeline,
    PROJECT_ROOT,
    REPORTS_DIR,
)

TICKERS_CSV = PROJECT_ROOT / "config" / "tickers.csv"
OPT_CONFIG = PROJECT_ROOT / "configs" / "optimization_config.yaml"

# Default representative subset for a quick, complete demonstration.
# Tuning all 50 is supported with --all (longer run).
DEFAULT_SUBSET = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "optimization.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("stockm.optimization_runner")


def load_tickers(csv_path: Path) -> list[str]:
    symbols: list[str] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            s = row["Symbol"].strip()
            if s:
                symbols.append(s)
    return symbols


def load_config(path: Path) -> dict:
    if not path.exists():
        logger.warning("Config %s not found; using in-code defaults.", path)
        return {}
    return yaml.safe_load(open(path, encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cfg = load_config(OPT_CONFIG)

    search = cfg.get("search", {})
    cvcfg = cfg.get("cv", {})
    runtime = cfg.get("runtime", {})
    models = cfg.get("models") or None

    pipeline = OptimizationPipeline(
        target_col=cfg.get("target_col", "target_next_return"),
        method=search.get("method", "random"),
        n_iter=int(search.get("n_iter", 15)),
        cv_splits=int(cvcfg.get("n_splits", 3)),
        cv_method=cvcfg.get("method", "timeseries"),
        scoring=search.get("scoring", "neg_root_mean_squared_error"),
        n_jobs=int(runtime.get("n_jobs", -1)),
        random_state=int(runtime.get("random_state", 42)),
        models=models,
    )

    logger.info(
        "Optimization run | method=%s n_iter=%d | cv=%s/%d | bayesian_available=%s",
        pipeline.method, pipeline.n_iter, pipeline.cv_method, pipeline.cv_splits,
        has_optuna(),
    )

    # Choose the ticker universe.
    if "--all" in argv:
        argv = [a for a in argv if a != "--all"]
        tickers = load_tickers(TICKERS_CSV)
    elif argv:
        tickers = [t.strip() for t in argv if t.strip()]
    else:
        tickers = DEFAULT_SUBSET
        logger.info("Using default subset %s (pass --all for every ticker).", tickers)

    summaries, failures = [], []
    for t in tickers:
        try:
            s = pipeline.run(t)
            summaries.append(s)
            logger.info(
                "  %-16s best_tuned=%-18s cv_rmse=%.6f deploy=%s dir_acc=%.4f stable=%s",
                t, s["best_tuned_model"], s["best_cv_rmse"], s["deploy"],
                s["directional_accuracy"], s["stable_across_regimes"],
            )
        except FileNotFoundError as e:
            failures.append((t, str(e)))
            logger.error("  %-16s SKIP (no prepared dataset)", t)
        except Exception as e:  # noqa: BLE001
            failures.append((t, repr(e)))
            logger.exception("  %-16s FAILED: %s", t, e)

    # ---- Global leaderboard ---------------------------------------------
    if summaries:
        leaderboard = sorted(
            summaries, key=lambda s: s["best_cv_rmse"] if s["best_cv_rmse"] is not None else 1e9
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "global_optimization_leaderboard.json").write_text(
            json.dumps(leaderboard, indent=2, default=str), encoding="utf-8"
        )
        logger.info("\n=== Global optimization leaderboard (by CV RMSE) ===")
        for i, s in enumerate(leaderboard, 1):
            logger.info(
                "  %2d. %-16s best=%-18s cv_rmse=%.6f deploy=%s dir_acc=%.4f",
                i, s["symbol"], s["best_tuned_model"], s["best_cv_rmse"],
                s["deploy"], s["directional_accuracy"],
            )

    logger.info(
        "\nDone. %d/%d succeeded | %d failure(s).", len(summaries), len(tickers), len(failures),
    )
    for sym, err in failures:
        logger.info("  FAILED %s: %s", sym, err)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
