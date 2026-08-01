"""
StockM v1.0 - Phase 8, Lesson 13
Save Results & Reproducibility Archive
======================================

Single Responsibility: persist EVERY artifact a backtest produces into a
structured, reproducible archive. Portfolio history, executed trades, metrics,
charts, configuration, and model version - all saved together so a future run
can be verified bit-for-bit against today's numbers.

Why a dedicated saver (not scattered writes)
--------------------------------------------
Reproducibility is a property of the WHOLE bundle, not individual files. A
metrics JSON without its config is uninterpretable; trades without the equity
curve can't be re-verified; a chart without the model version can't be
attributed. The saver writes a coherent bundle with a manifest tying every
artifact to the run that produced it. Given the same predictions + config +
model version, the numbers must match (the reproducibility contract).

Folder layout (per the Phase 8 roadmap)
---------------------------------------
    backtests/
        results/        - one bundle per run (the manifest + everything)
        equity_curves/  - date-indexed equity Series as CSV (fast to reload)
        reports/        - Markdown + JSON reports (Lesson 12)
        metrics/        - metrics dicts as JSON (for cross-run leaderboards)
        trade_logs/     - executed trades as CSV

A single run writes to ALL five, linked by a shared run_id. The bundle under
results/<run_id>/ is the canonical archive; the other four are convenience
views (a metrics leaderboard scans metrics/; a trade-log audit scans
trade_logs/) that point back to the bundle.

Reproducibility contract
------------------------
Given (predictions, config, model_version), re-running must produce numbers
matching the saved metrics within float tolerance. The saver records the
exact inputs (config hash, model version, n_bars, n_trades) so a future
verification can detect drift: if the metrics don't match the saved ones for
the same config+model, something in the pipeline changed and the old result
is stale. This is MLOps discipline: a result without provenance is noise.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.engine import BacktestResult

logger = logging.getLogger("stockm.backtesting.results_saver")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTESTS_DIR = PROJECT_ROOT / "backtests"
RESULTS_DIR = BACKTESTS_DIR / "results"
EQUITY_CURVES_DIR = BACKTESTS_DIR / "equity_curves"
REPORTS_DIR = BACKTESTS_DIR / "reports"
METRICS_DIR = BACKTESTS_DIR / "metrics"
TRADE_LOGS_DIR = BACKTESTS_DIR / "trade_logs"


def _ensure_dirs() -> None:
    """Create the five-folder layout (idempotent)."""
    for d in (RESULTS_DIR, EQUITY_CURVES_DIR, REPORTS_DIR, METRICS_DIR, TRADE_LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _config_hash(config: dict) -> str:
    """A short deterministic hash of the run config (for reproducibility drift checks).

    SHA-256 of the canonical JSON; first 12 chars is enough to spot a config
    change (a different config -> a different hash -> a flag that old results
    may be stale).
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _run_id(strategy_name: str, config: dict) -> str:
    """Build a unique run id: <strategy>_<date>_<cfghash>.

    Date + config hash disambiguates same-strategy runs on different days or
    with different configs. No randomness (Math.random/uuid4 would break the
    reproducibility contract - the same inputs must yield the same id).
    """
    today = date.today().isoformat()
    return f"{strategy_name}_{today}_{_config_hash(config)}"


class ResultsSaver:
    """Persist a BacktestResult (and optional comparison/stress) as a full archive.

    Args:
        base_dir: Root for the five-folder layout. Default backtests/.
        model_version: The model that produced the predictions (for provenance).
                       If None, read from the result's metadata.
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        model_version: str | None = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else BACKTESTS_DIR
        self.results_dir = self.base_dir / "results"
        self.equity_dir = self.base_dir / "equity_curves"
        self.reports_dir = self.base_dir / "reports"
        self.metrics_dir = self.base_dir / "metrics"
        self.trades_dir = self.base_dir / "trade_logs"
        self.model_version = model_version
        for d in (self.results_dir, self.equity_dir, self.reports_dir,
                  self.metrics_dir, self.trades_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, result: BacktestResult) -> Path:
        """Save a single BacktestResult as a full reproducible bundle.

        Writes:
            results/<run_id>/manifest.json     - provenance + file index
            results/<run_id>/config.json       - the exact run config
            results/<run_id>/metrics.json      - the metrics dict
            results/<run_id>/equity_curve.csv  - date,equity
            results/<run_id>/trades.csv        - the full trade log
        Plus convenience views:
            equity_curves/<run_id>.csv         - same equity (fast reload)
            metrics/<run_id>.json              - same metrics (leaderboard scan)
            trade_logs/<run_id>.csv            - same trades (audit scan)

        Args:
            result: The BacktestResult to archive.

        Returns:
            Path to the bundle's manifest.json (the canonical archive entry).
        """
        model_ver = self.model_version or result.metadata.get(
            "model_version", result.metadata.get("source", "unknown"))
        run_id = _run_id(result.strategy_name, result.config)
        bundle = self.results_dir / run_id
        bundle.mkdir(parents=True, exist_ok=True)

        # --- Core artifacts ---
        config_path = bundle / "config.json"
        config_path.write_text(json.dumps(result.config, indent=2, default=str), encoding="utf-8")

        metrics_path = bundle / "metrics.json"
        metrics_path.write_text(json.dumps(result.metrics, indent=2, default=str), encoding="utf-8")

        equity_path = bundle / "equity_curve.csv"
        eq_df = result.equity_curve.to_frame(name="equity")
        eq_df.index.name = "date"
        eq_df.to_csv(equity_path)

        trades_path = bundle / "trades.csv"
        self._write_trades_csv(result.trades, trades_path)

        # --- Convenience views (point back to the bundle) ---
        self._copy(equity_path, self.equity_dir / f"{run_id}.csv")
        self._copy(metrics_path, self.metrics_dir / f"{run_id}.json")
        self._copy(trades_path, self.trades_dir / f"{run_id}.csv")

        # --- Manifest (provenance + file index) ---
        manifest = {
            "run_id": run_id,
            "strategy_name": result.strategy_name,
            "symbols": result.symbols,
            "model_version": model_ver,
            "stockm_version": result.metadata.get("stockm_version", "unknown"),
            "config_hash": _config_hash(result.config),
            "created": date.today().isoformat(),
            "n_bars": int(result.metadata.get("n_bars", len(result.equity_curve))),
            "n_trades": len(result.trades),
            "total_return": result.metrics["total_return"],
            "files": {
                "config": str(config_path.relative_to(self.base_dir)),
                "metrics": str(metrics_path.relative_to(self.base_dir)),
                "equity_curve": str(equity_path.relative_to(self.base_dir)),
                "trades": str(trades_path.relative_to(self.base_dir)),
            },
        }
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        logger.info(
            "saved run '%s' -> %s (model=%s, ret=%.2f%%, %d trades)",
            run_id, bundle.relative_to(self.base_dir), model_ver,
            result.metrics["total_return"] * 100, len(result.trades),
        )
        return manifest_path

    def save_comparison(self, comparison, strategy_results: list[BacktestResult] | None = None) -> list[Path]:
        """Save a comparison's individual results + a comparison manifest.

        Each strategy's BacktestResult is saved as its own bundle (via
        :meth:`save`), and a comparison manifest links them with the ranking
        and recommendation. This preserves per-strategy reproducibility while
        keeping the comparison view.

        Args:
            comparison:        A ComparisonResult (Lesson 10).
            strategy_results:  The BacktestResults behind it (from comparison.results).
                               If None, read from comparison.results.

        Returns:
            List of manifest paths (one per strategy + the comparison manifest).
        """
        results = strategy_results if strategy_results is not None else comparison.results
        manifests = [self.save(r) for r in results]
        # Comparison manifest.
        cmp_manifest = {
            "type": "comparison",
            "created": date.today().isoformat(),
            "recommended": comparison.recommended,
            "recommended_beats_benchmark": comparison.recommended_beats_benchmark,
            "benchmark_return": comparison.benchmark_return,
            "ranking": [r.__dict__ for r in comparison.rows],
            "run_ids": [r.strategy_name + "_" + date.today().isoformat() + "_" + _config_hash(r.config)
                        for r in results],
        }
        cmp_path = self.results_dir / f"comparison_{date.today().isoformat()}.json"
        cmp_path.write_text(json.dumps(cmp_manifest, indent=2, default=str), encoding="utf-8")
        manifests.append(cmp_path)
        logger.info("saved comparison manifest -> %s", cmp_path.relative_to(self.base_dir))
        return manifests

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _copy(src: Path, dst: Path) -> None:
        """Copy a file (the convenience views are duplicates of bundle files)."""
        dst.write_bytes(src.read_bytes())

    @staticmethod
    def _write_trades_csv(trades: list, path: Path) -> None:
        """Write the trade log as CSV (one row per executed Trade)."""
        if not trades:
            path.write_text("date,symbol,action,quantity,price,value,cost,reason\n", encoding="utf-8")
            return
        rows = []
        for t in trades:
            reason = ""
            if t.signal is not None:
                reason = t.signal.reason or ""
            rows.append({
                "date": t.date, "symbol": t.symbol, "action": t.action.value,
                "quantity": t.quantity, "price": t.price, "value": t.value,
                "cost": t.cost, "reason": reason,
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    # ----------------------------------------------------------- reload + verify

    def load_metrics(self, run_id: str) -> dict[str, float]:
        """Reload a saved metrics dict (for cross-run leaderboards)."""
        p = self.metrics_dir / f"{run_id}.json"
        return json.loads(p.read_text(encoding="utf-8"))

    def load_equity_curve(self, run_id: str) -> pd.Series:
        """Reload a saved equity curve (date-indexed Series)."""
        p = self.equity_dir / f"{run_id}.csv"
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df["equity"]

    def verify_reproducibility(self, run_id: str, current_result: BacktestResult, tol: float = 1e-6) -> dict[str, Any]:
        """Check whether a re-run matches a saved bundle (the reproducibility contract).

        Compares the current run's metrics against the saved ones. If they
        match within tolerance, the pipeline is reproducible. If not, something
        changed (config, model, data) and the old result is stale - this is the
        MLOps drift detector.

        Args:
            run_id:          The saved run to compare against.
            current_result:  A freshly-computed BacktestResult.
            tol:             Float tolerance per metric.

        Returns:
            Dict with per-metric match flags + an overall 'reproducible' bool.
        """
        saved = self.load_metrics(run_id)
        out: dict[str, Any] = {"run_id": run_id, "matches": {}, "reproducible": True}
        for k, v in saved.items():
            cur = current_result.metrics.get(k)
            match = cur is not None and abs(float(v) - float(cur)) <= tol
            out["matches"][k] = {"saved": v, "current": cur, "match": match}
            if not match:
                out["reproducible"] = False
        return out

    def leaderboard(self) -> pd.DataFrame:
        """Scan metrics/ and return a cross-run leaderboard (for strategy selection).

        One row per saved run, sorted by total_return descending. The fast way
        to answer "which run was best?" without reloading full equity curves.
        """
        rows = []
        for p in sorted(self.metrics_dir.glob("*.json")):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                rows.append({"run_id": p.stem, **m})
            except Exception:  # noqa: BLE001
                continue
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "total_return" in df.columns:
            df = df.sort_values("total_return", ascending=False)
        return df.reset_index(drop=True)
