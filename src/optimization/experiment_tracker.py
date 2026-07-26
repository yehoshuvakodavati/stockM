"""
StockM v1.0 - Phase 6, Lesson 11
Experiment Tracker
==================

Logs every optimization experiment so any run is reproducible and
comparable months later. Without this, "I tuned it once and it worked" is
untraceable - you can't re-run, can't diff configs, can't audit what went
to production.

Recorded per experiment
-----------------------
  experiment_id   stable unique key (symbol + model + method + short hash)
  timestamp       when it ran (ISO UTC)
  symbol, model_name, method (grid/random/bayesian)
  hyperparameters the searched config (JSON)
  dataset_version ticker + train row count + date range (the exact data)
  cv              method + n_splits
  metrics         validation (CV) + test metrics + directional accuracy
  training_time_s wall-clock for the search
  best            whether this is the deployed model for the ticker
  seed            random_state for reproducibility

Storage: one CSV (experiments/optimization_runs.csv) for tabular
comparison across runs, plus one JSON per ticker with full detail. The CSV
is append-only so successive runs accumulate a history.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("stockm.optimization.tracker")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_CSV = EXPERIMENTS_DIR / "optimization_runs.csv"

# Flat CSV column order. Hyperparameters are JSON-encoded into one cell so
# the CSV stays rectangular.
CSV_COLUMNS = [
    "experiment_id", "timestamp", "symbol", "model_name", "method",
    "hyperparameters", "dataset_version", "cv_method", "n_splits",
    "val_rmse", "val_r2", "test_rmse", "test_r2", "directional_accuracy",
    "beats_baseline", "training_time_s", "seed", "best",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_experiment_id(symbol: str, model_name: str, method: str, params: dict) -> str:
    """Stable id: deterministic over identical inputs, unique otherwise."""
    h = hashlib.md5(
        f"{symbol}|{model_name}|{method}|{json.dumps(params, sort_keys=True)}".encode()
    ).hexdigest()[:10]
    return f"{symbol.replace('.', '_')}_{model_name}_{method}_{h}"


class ExperimentTracker:
    """Append-only experiment log (CSV) + per-ticker JSON detail."""

    def __init__(self, experiments_dir: Path | None = None) -> None:
        self.experiments_dir = experiments_dir or EXPERIMENTS_DIR
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        path = self.experiments_dir / "optimization_runs.csv"
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def log(self, record: dict[str, Any]) -> str:
        """Append one experiment row to the CSV and save a ticker JSON.

        Args:
            record: dict with keys matching CSV_COLUMNS (hyperparameters,
                dataset_version may be dicts -> JSON-encoded).

        Returns:
            The experiment_id.
        """
        exp_id = record.get("experiment_id") or make_experiment_id(
            record["symbol"], record["model_name"], record["method"],
            record.get("hyperparameters", {}),
        )
        record = {**record, "experiment_id": exp_id}

        # Flatten nested fields to JSON strings for the CSV cell.
        row = {c: record.get(c, "") for c in CSV_COLUMNS}
        row["experiment_id"] = exp_id
        for key in ("hyperparameters", "dataset_version"):
            if isinstance(row.get(key), dict):
                row[key] = json.dumps(row[key], sort_keys=True)
        for key in ("val_rmse", "val_r2", "test_rmse", "test_r2",
                    "directional_accuracy", "training_time_s"):
            v = row.get(key)
            if isinstance(v, float) and v != v:
                row[key] = ""  # NaN -> empty cell (cleaner CSV)
        row["best"] = "true" if record.get("best") else "false"

        with open(self.experiments_dir / "optimization_runs.csv",
                  "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[c] for c in CSV_COLUMNS])

        # Per-ticker JSON with full detail (richer than the flat CSV).
        jp = self.experiments_dir / f"{record['symbol'].replace('.', '_')}_optimization_experiments.json"
        existing = []
        if jp.exists():
            try:
                existing = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append({**record, "timestamp": record.get("timestamp", _now_iso())})
        jp.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        logger.info("logged experiment %s", exp_id)
        return exp_id

    def load_history(self) -> list[dict[str, Any]]:
        """Return all logged experiments (parsed from the CSV)."""
        path = self.experiments_dir / "optimization_runs.csv"
        if not path.exists():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
