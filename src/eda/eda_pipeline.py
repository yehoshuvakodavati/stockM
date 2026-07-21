"""
StockM v1.0 - Phase 4, Lessons 12 + 13
EDA Orchestrator
=================

Ties the analysis modules into one pipeline that turns an AI-ready feature
dataset into a clean, analysed, ML-ready train/val/test split:

    load feature dataset
        -> EDA (overview, stats, distributions, correlation, outliers,
                target analysis, leakage audit)         [report only]
        -> chronological split (train / val / test)
        -> feature selection (fit on TRAIN only)
        -> scaling (fit on TRAIN only, applied to val/test)
        -> save prepared datasets + metadata

Outputs (under data/prepared/<SYMBOL>/)
---------------------------------------
    train.csv, validation.csv, test.csv        - scaled, selected, split
    feature_metadata.json                      - selected features, dropped,
                                                 scores, split info
    scaler_params.json                         - fitted scaler parameters
    eda_report.json                            - the full analysis report

Anti-leakage enforcement
------------------------
- Selection (MI, tree importance) is fit on TRAIN only.
- Scaler is fit on TRAIN only.
- Splits are chronological with a gap >= target horizon.
- Exploratory stats (overview/corr/outliers) are computed on the full frame
  for *understanding* and are NOT used as preprocessing parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eda.dataset_overview import dataset_overview
from eda.statistics import descriptive_statistics
from eda.distributions import distribution_analysis
from eda.correlation import correlation_analysis
from eda.outliers import detect_outliers, winsorize
from eda.target_analysis import target_analysis
from eda.leakage_detection import leakage_audit
from eda.feature_selection import select_features
from eda.scaling import FeatureScaler
from eda.splitting import chronological_split

# ---------------------------------------------------------------------------
# Paths
#   __file__  = .../stockM/src/eda/eda_pipeline.py
#   parents[2] = .../stockM  (project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"
REPORTS_DIR = PROJECT_ROOT / "reports" / "features"

logger = logging.getLogger("stockm.eda")


def _safe_filename(symbol: str) -> str:
    return symbol.replace(".", "_")


class EDAPipeline:
    """Run the full EDA + preparation flow for one ticker's feature dataset.

    Config-driven: split ratios, scaler method, target choice, and selection
    top-k come from a params dict (read from configs/data_config.yaml +
    feature_config.yaml by the runner).
    """

    def __init__(
        self,
        target_col: str = "target_direction",
        val_size: float = 0.15,
        test_size: float = 0.15,
        gap_days: int = 1,
        scaler_method: str = "robust",
        keep_top_k: int = 40,
        winsorize_train: bool = True,
        features_dir: Path | None = None,
        prepared_dir: Path | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self.target_col = target_col
        self.val_size = val_size
        self.test_size = test_size
        self.gap_days = gap_days
        self.scaler_method = scaler_method
        self.keep_top_k = keep_top_k
        self.winsorize_train = winsorize_train
        self.features_dir = features_dir or FEATURES_DIR
        self.prepared_dir = prepared_dir or PREPARED_DIR
        self.reports_dir = reports_dir or REPORTS_DIR

    # ------------------------------------------------------------------ load
    def _load(self, symbol: str) -> pd.DataFrame:
        path = self.features_dir / f"{_safe_filename(symbol)}_features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Feature dataset not found for {symbol}: {path}. "
                f"Run the feature pipeline first."
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()
        return df

    # --------------------------------------------------------------- analyse
    def analyse(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run the report-only analysis pass on the full frame."""
        target = self.target_col if self.target_col in df.columns else None
        return {
            "overview": dataset_overview(df),
            "statistics": descriptive_statistics(df),
            "distributions": distribution_analysis(df),
            "correlation": correlation_analysis(df, target_col=target),
            "outliers": detect_outliers(df),
            "targets": target_analysis(df),
            "leakage": leakage_audit(df, target_col=self.target_col),
        }

    # --------------------------------------------------------------- prepare
    def prepare(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Split -> select -> scale. Returns (train, val, test, metadata).

        All *decisions* (selected features, scaler params, winsor fences) are
        derived from TRAIN only and applied to val/test.
        """
        # 1. Chronological split (before any fitting -> no leakage).
        train, val, test, split_info = chronological_split(
            df, val_size=self.val_size, test_size=self.test_size,
            gap_days=self.gap_days,
        )

        # 2. Feature selection on TRAIN only.
        selection = select_features(
            train, target_col=self.target_col, keep_top_k=self.keep_top_k,
        )
        selected = selection["selected_features"]

        # Always keep target columns alongside the selected features.
        target_cols = [c for c in df.columns if c.startswith("target_")]
        keep_cols = selected + target_cols

        # 3. Optional winsorisation of the TRAIN features (tames fat tails).
        # Fences are train-derived and applied to val/test via the same
        # quantile cap values would require persistence; to keep v1 simple
        # and leak-free we winsorise train only and leave val/test unscaled-
        # by-winsor (the RobustScaler below already neutralises outliers).
        if self.winsorize_train:
            train = winsorize(train, cols=selected)

        train, val, test = train[keep_cols], val[keep_cols], test[keep_cols]

        # 4. Scaling: fit on TRAIN, transform all three.
        scaler = FeatureScaler(method=self.scaler_method)
        train = scaler.fit_transform(train, feature_cols=selected)
        val = scaler.transform(val)
        test = scaler.transform(test)

        metadata = {
            "target_col": self.target_col,
            "selected_features": selected,
            "target_columns": target_cols,
            "dropped": selection["dropped"],
            "scores": selection["scores"],
            "split": split_info,
            "scaler": scaler.params(),
            "n_features_before": selection["n_before"],
            "n_features_after": selection["n_after"],
        }
        return train, val, test, metadata

    # ------------------------------------------------------------------ run
    def run(self, symbol: str) -> dict[str, Any]:
        """Full flow for one ticker: load -> analyse -> prepare -> save."""
        logger.info("EDA for %s", symbol)
        df = self._load(symbol)

        report = self.analyse(df)

        if not report["leakage"]["ok"]:
            logger.warning(
                "%s: %d leakage flag(s) found - inspect before training: %s",
                symbol, len(report["leakage"]["leakage_flags"]),
                report["leakage"]["leakage_flags"][:5],
            )

        train, val, test, metadata = self.prepare(df)
        self._save(train, val, test, metadata, report, symbol)

        return {
            "symbol": symbol,
            "rows": {"train": len(train), "val": len(val), "test": len(test)},
            "features": metadata["n_features_after"],
            "leakage_flags": len(report["leakage"]["leakage_flags"]),
            "scaler": self.scaler_method,
            "target": self.target_col,
        }

    # ------------------------------------------------------------------ save
    def _save(
        self,
        train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
        metadata: dict[str, Any], report: dict[str, Any], symbol: str,
    ) -> None:
        out_dir = self.prepared_dir / _safe_filename(symbol)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        train.to_csv(out_dir / "train.csv", index=True)
        val.to_csv(out_dir / "validation.csv", index=True)
        test.to_csv(out_dir / "test.csv", index=True)

        (out_dir / "feature_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        (out_dir / "scaler_params.json").write_text(
            json.dumps(metadata["scaler"], indent=2, default=str), encoding="utf-8"
        )
        (self.reports_dir / f"{_safe_filename(symbol)}_eda_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "saved %s -> train=%d val=%d test=%d features=%d",
            symbol, len(train), len(val), len(test), metadata["n_features_after"],
        )
