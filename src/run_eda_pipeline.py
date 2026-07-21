"""
StockM v1.0 - Phase 4: Dataset Preparation & EDA
Entry point for the EDA + dataset-preparation batch pipeline.

Flow:
    for each ticker with a feature dataset:
        load data/processed/features/<SYMBOL>_features.csv
        -> EDA report (overview/stats/distributions/correlation/outliers/
                       targets/leakage)
        -> chronological split (train/val/test, gap >= horizon)
        -> feature selection (fit on TRAIN)
        -> scaling (fit on TRAIN, applied to val/test)
        -> save data/prepared/<SYMBOL>/{train,validation,test}.csv
           + feature_metadata.json + scaler_params.json
        -> save reports/features/<SYMBOL>_eda_report.json

Config is read from configs/data_config.yaml (splits, target) and
configs/feature_config.yaml (scaler). Run from the project root:

    python src/run_eda_pipeline.py
    python src/run_eda_pipeline.py RELIANCE.NS
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda.eda_pipeline import EDAPipeline, PROJECT_ROOT

TICKERS_CSV = PROJECT_ROOT / "config" / "tickers.csv"
DATA_CONFIG = PROJECT_ROOT / "configs" / "data_config.yaml"
FEATURE_CONFIG = PROJECT_ROOT / "configs" / "feature_config.yaml"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "eda.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("stockm.eda_runner")


def load_tickers(csv_path: Path) -> list[str]:
    symbols: list[str] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            s = row["Symbol"].strip()
            if s:
                symbols.append(s)
    return symbols


def load_config() -> dict:
    """Merge the split/target config (data_config) + scaler (feature_config)."""
    cfg: dict = {}
    if DATA_CONFIG.exists():
        d = yaml.safe_load(open(DATA_CONFIG, encoding="utf-8")) or {}
        splits = d.get("splits", {})
        target = d.get("target", {})
        cfg["val_size"] = float(splits.get("val_size", 0.15))
        cfg["test_size"] = float(splits.get("test_size", 0.15))
        cfg["gap_days"] = int(splits.get("gap_days", 1))
        # target.type binary_direction -> use target_direction column.
        cfg["target_col"] = "target_direction" if target.get(
            "type", "binary_direction") == "binary_direction" else "target_signal"
    if FEATURE_CONFIG.exists():
        f = yaml.safe_load(open(FEATURE_CONFIG, encoding="utf-8")) or {}
        scaling = f.get("scaling", {})
        cfg["scaler_method"] = scaling.get("method", "robust")
    return cfg


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cfg = load_config()

    pipeline = EDAPipeline(
        target_col=cfg.get("target_col", "target_direction"),
        val_size=cfg.get("val_size", 0.15),
        test_size=cfg.get("test_size", 0.15),
        gap_days=cfg.get("gap_days", 1),
        scaler_method=cfg.get("scaler_method", "robust"),
        keep_top_k=40,
    )

    tickers = [t.strip() for t in argv if t.strip()] or load_tickers(TICKERS_CSV)
    logger.info(
        "EDA run: %d ticker(s) | target=%s | split=%.2f/%.2f gap=%d | scaler=%s",
        len(tickers), pipeline.target_col, pipeline.val_size, pipeline.test_size,
        pipeline.gap_days, pipeline.scaler_method,
    )

    summaries, failures = [], []
    for t in tickers:
        try:
            s = pipeline.run(t)
            summaries.append(s)
            logger.info(
                "  %-16s train=%-5d val=%-5d test=%-5d feats=%-3d leak_flags=%d",
                t, s["rows"]["train"], s["rows"]["val"], s["rows"]["test"],
                s["features"], s["leakage_flags"],
            )
        except FileNotFoundError as e:
            failures.append((t, str(e)))
            logger.error("  %-16s SKIP (no feature dataset)", t)
        except Exception as e:  # noqa: BLE001
            failures.append((t, repr(e)))
            logger.exception("  %-16s FAILED: %s", t, e)

    total_train = sum(s["rows"]["train"] for s in summaries)
    logger.info(
        "\nDone. %d/%d succeeded | %d total train-rows prepared | %d failure(s).",
        len(summaries), len(tickers), total_train, len(failures),
    )
    for sym, err in failures:
        logger.info("  FAILED %s: %s", sym, err)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
