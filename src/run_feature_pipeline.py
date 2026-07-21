"""
StockM v1.0 - Phase 3: Feature Engineering
Entry point for the feature-engineering batch pipeline.

Flow:
    read config/tickers.csv
        -> for each symbol:
              load data/raw/<SYMBOL>.csv
              -> build_features (price, trend, momentum, volatility,
                                 volume, time, lag, rolling, target)
              -> handle missing values
              -> validate
              -> save data/processed/features/<SYMBOL>_features.csv
                 + data/processed/metadata/<SYMBOL>_feature_report.json

Parameters are read from configs/feature_config.yaml (ohlcv block) so the
whole project shares one source of truth - no hard-coded windows here.

Run from the project root:
    python src/run_feature_pipeline.py                 # all tickers
    python src/run_feature_pipeline.py RELIANCE.NS      # one ticker
    python src/run_feature_pipeline.py RELIANCE.NS INFY.NS   # a subset
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import yaml

# Make the 'feature_engineering' package importable when running
# `python src/run_feature_pipeline.py` (mirrors src/main.py's convention:
# Python puts the script's own dir, src/, on sys.path automatically).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering.feature_pipeline import FeatureEngine, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TICKERS_CSV = PROJECT_ROOT / "config" / "tickers.csv"
FEATURE_CONFIG = PROJECT_ROOT / "configs" / "feature_config.yaml"

# ---------------------------------------------------------------------------
# Logging - console + a rotating file under logs/
# ---------------------------------------------------------------------------
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "feature_engineering.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("stockm.runner")


def load_tickers(csv_path: Path) -> list[str]:
    """Read ticker symbols from the one-column 'Symbol' CSV (skip blanks)."""
    symbols: list[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["Symbol"].strip()
            if symbol:
                symbols.append(symbol)
    return symbols


def load_params(config_path: Path) -> dict:
    """Read the `ohlcv` block from feature_config.yaml into a flat params dict."""
    if not config_path.exists():
        logger.warning("Config %s not found; using in-code defaults.", config_path)
        return {}
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    params = cfg.get("ohlcv", {}) or {}

    # The active-group gate: only compute when `ohlcv` is listed under
    # active_groups. If it isn't, we still proceed (v1 requires it) but warn.
    active = cfg.get("active_groups", [])
    if "ohlcv" not in active:
        logger.warning(
            "'ohlcv' not under active_groups in %s; proceeding anyway (v1 core).",
            config_path.name,
        )
    return params


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    params = load_params(FEATURE_CONFIG)
    missing_strategy = params.pop("missing_strategy", "drop")
    corr_threshold = float(params.pop("corr_threshold", 0.95))

    # Tick the one non-ohlcv-block knob into the engine via a side channel:
    # the validator threshold. We pass it through params under a reserved key.
    engine = FeatureEngine(
        params=params,
        missing_strategy=missing_strategy,
    )

    # Choose ticker universe: CLI args override the CSV when provided.
    if argv:
        tickers = [t.strip() for t in argv if t.strip()]
    else:
        tickers = load_tickers(TICKERS_CSV)

    logger.info(
        "Feature engineering run: %d ticker(s), missing_strategy=%s",
        len(tickers), missing_strategy,
    )

    summaries: list[dict] = []
    failures: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            summary = engine.process_ticker(ticker)
            summaries.append(summary)
            logger.info(
                "  %-16s rows=%-5d cols=%-3d dropped=%-5d ok=%s corr_pairs=%d",
                ticker,
                summary["feature_rows"],
                summary["feature_cols"],
                summary["rows_dropped"],
                summary["validation_ok"],
                summary["correlated_pairs"],
            )
        except FileNotFoundError as e:
            failures.append((ticker, str(e)))
            logger.error("  %-16s SKIP (raw data missing)", ticker)
        except Exception as e:  # noqa: BLE001 - batch run must continue
            failures.append((ticker, repr(e)))
            logger.exception("  %-16s FAILED: %s", ticker, e)

    # ---- Run summary ------------------------------------------------------
    total_rows = sum(s["feature_rows"] for s in summaries)
    logger.info(
        "\nDone. %d/%d tickers succeeded | %d total feature-rows written | "
        "%d failure(s).",
        len(summaries), len(tickers), total_rows, len(failures),
    )
    if failures:
        for sym, err in failures:
            logger.info("  FAILED %s: %s", sym, err)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
