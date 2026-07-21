"""
StockM v1.0 - Phase 5: Baseline Model Training
Entry point for the ML baseline batch pipeline.

Flow (per ticker):
    load data/prepared/<SYMBOL>/{train,validation,test}.csv
        -> split X/y (leakage-safe)
        -> train every available baseline (Linear, DT, RF, GBM, [+XGB/LGBM])
        -> evaluate on val (selection) + test (honest report)
        -> feature importance + error analysis
        -> save all models + versioned metadata; mark best as deployed
        -> save per-ticker report under reports/training/

Finally, aggregates a global leaderboard across all tickers.

Config is read from configs/model_config.yaml (baseline hyperparameters).
Run from the project root:
    python src/run_baseline_training.py
    python src/run_baseline_training.py RELIANCE.NS
    python src/run_baseline_training.py RELIANCE.NS INFY.NS
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.baseline_models import DEFAULT_PARAMS, available_models
from models.training_pipeline import BaselineTrainer, PROJECT_ROOT, REPORTS_DIR

TICKERS_CSV = PROJECT_ROOT / "config" / "tickers.csv"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model_config.yaml"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "baseline_training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("stockm.baseline_runner")


def load_tickers(csv_path: Path) -> list[str]:
    symbols: list[str] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            s = row["Symbol"].strip()
            if s:
                symbols.append(s)
    return symbols


def load_model_overrides(config_path: Path) -> dict[str, dict]:
    """Read baseline hyperparameters from model_config.yaml.

    Maps the config's baseline blocks (xgboost_baseline, etc.) onto our
    model names. Missing/unused params are ignored; in-code DEFAULT_PARAMS
    apply otherwise.
    """
    if not config_path.exists():
        return {}
    cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    overrides: dict[str, dict] = {}

    xgb_block = cfg.get("xgboost_baseline", {})
    if xgb_block:
        # Reuse the same sensible params for both xgboost and the sklearn booster.
        params = {k: v for k, v in xgb_block.items() if k not in ("family", "type")}
        if "n_estimators" in params:
            overrides.setdefault("xgboost", {}).update(params)
            overrides.setdefault("gradient_boosting", {}).update(params)
    return overrides


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    overrides = load_model_overrides(MODEL_CONFIG)

    trainer = BaselineTrainer(
        target_col="target_next_return",
        model_overrides=overrides,
    )

    logger.info(
        "Baseline run | models=%s | target=%s",
        available_models(), trainer.target_col,
    )

    tickers = [t.strip() for t in argv if t.strip()] or load_tickers(TICKERS_CSV)
    logger.info("Training %d ticker(s)...", len(tickers))

    summaries, failures = [], []
    for t in tickers:
        try:
            s = trainer.run(t)
            summaries.append(s)
            logger.info(
                "  %-16s best=%-18s val_rmse=%.6f test_rmse=%.6f dir_acc=%.4f beats_naive=%s",
                t, s["best_model"], s["val_rmse"], s["test_rmse"],
                s["directional_accuracy"], s["beats_naive"],
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
            summaries, key=lambda s: s["val_rmse"] if s["val_rmse"] is not None else 1e9
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "global_leaderboard.json").write_text(
            json.dumps(leaderboard, indent=2, default=str), encoding="utf-8"
        )
        logger.info("\n=== Global leaderboard (by validation RMSE) ===")
        for i, s in enumerate(leaderboard, 1):
            logger.info(
                "  %2d. %-16s best=%-18s val_rmse=%.6f test_rmse=%.6f dir_acc=%.4f",
                i, s["symbol"], s["best_model"], s["val_rmse"], s["test_rmse"],
                s["directional_accuracy"],
            )

    logger.info(
        "\nDone. %d/%d succeeded | %d failure(s).", len(summaries), len(tickers), len(failures),
    )
    for sym, err in failures:
        logger.info("  FAILED %s: %s", sym, err)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
