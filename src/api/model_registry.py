"""
StockM v1.0 - Phase 9, Lesson 4
Model Registry
==============

Single source of truth for which models are deployed and how to load them.
Discovers models from the three existing stores — baseline ML (models/
saved_models), optimized ML (models/optimized), and DL (models/dl_models) —
loads their metadata (cheap), and lazily loads the heavy model object only when
first requested for prediction. This keeps startup fast (metadata-only) while
making the first prediction for a symbol pay a one-time load cost.

Why a registry (not "load on demand by path")
---------------------------------------------
In production you must answer: which models exist? which version is deployed?
what features does each need? A registry answers these without touching the
model objects. It also centralizes the resolution rule (optimized > baseline;
ML and DL kept side by side), so the prediction service never hard-codes paths.
Swapping the model store (e.g. an MLflow registry) changes only this file.

Resolution rule (mirrors models.prediction.load_model)
------------------------------------------------------
For a symbol, the "active" ML model is:
    1. optimized/<SYM>/best_optimized.json  if it exists (Phase 6 tuned), else
    2. saved_models/<SYM>/best_model.json   (Phase 5 baseline).
DL models live in models/dl_models/<SYM>/best_dl_model.json and are a parallel
track (the API can serve either family).

Lazy loading
------------
``load_all()`` reads only JSON metadata (fast: ~50 symbols in milliseconds).
The actual model object (joblib/torch) is loaded on first ``get_model(symbol)``
and cached in ``_loaded``. This separates "what's deployed" (startup) from
"load the weights" (first request) — critical for cold-start time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("stockm.api.registry")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
OPTIMIZED_DIR = PROJECT_ROOT / "models" / "optimized"
DL_MODELS_DIR = PROJECT_ROOT / "models" / "dl_models"


@dataclass
class ModelEntry:
    """Metadata for one deployed model (the registry's unit of record).

    Attributes:
        symbol:        Ticker, e.g. "RELIANCE.NS".
        model_name:    The deployed model's name (from best_*.json).
        model_type:    "ml" or "dl".
        source:        "baseline" | "optimized" | "dl".
        model_version: Project version string (from metadata).
        target_col:    Target the model predicts (e.g. target_next_return).
        feature_names: The feature list the model expects (for validation).
        metrics:       Validation/test metrics (for the /models endpoint).
        metadata:      The full metadata dict (provenance for the report).
    """

    symbol: str
    model_name: str
    model_type: str  # "ml" | "dl"
    source: str      # "baseline" | "optimized" | "dl"
    model_version: str = "unknown"
    target_col: str = "target_next_return"
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    arch_type: str | None = None  # DL only (lstm/gru/cnn/transformer)


class ModelRegistry:
    """Discover, index, and lazily load deployed models.

    Lifecycle:
        registry = ModelRegistry()
        registry.load_all()                      # fast: metadata only
        entry = registry.get_entry("RELIANCE.NS")
        model = registry.get_model("RELIANCE.NS")  # lazy: loads weights once
    """

    def __init__(self) -> None:
        # symbol -> ModelEntry (metadata, loaded at startup)
        self._entries: dict[str, ModelEntry] = {}
        # symbol -> loaded model object (lazy, cached after first get_model)
        self._loaded: dict[str, Any] = {}

    # -------------------------------------------------- discovery / loading

    def load_all(self) -> None:
        """Discover all deployed models and load their metadata (not weights).

        Scans the three stores and builds the entry index. A missing store is
        not an error (e.g. a fresh checkout with no optimized models). The
        resolution rule means a symbol with both baseline and optimized models
        appears once, as "optimized".
        """
        self._entries.clear()
        self._loaded.clear()

        # 1) Baseline ML (lowest priority — may be overridden by optimized).
        self._discover_ml(SAVED_MODELS_DIR, source="baseline")
        # 2) Optimized ML (overrides baseline for the same symbol).
        self._discover_ml(OPTIMIZED_DIR, source="optimized")
        # 3) DL (parallel track; keyed under symbol+"_dl" to coexist with ML).
        self._discover_dl()

        logger.info(
            "registry loaded: %d ML symbols + %d DL symbols",
            sum(1 for e in self._entries.values() if e.model_type == "ml"),
            sum(1 for e in self._entries.values() if e.model_type == "dl"),
        )

    def _discover_ml(self, base_dir: Path, source: str) -> None:
        """Discover ML models (baseline or optimized) in one store."""
        if not base_dir.exists():
            return
        for sym_dir in base_dir.iterdir():
            if not sym_dir.is_dir():
                continue
            best = self._read_best(sym_dir, source)
            if best is None:
                continue
            model_name, symbol, meta = best
            self._entries[symbol] = ModelEntry(
                symbol=symbol, model_name=model_name, model_type="ml", source=source,
                model_version=meta.get("model_version", "unknown"),
                target_col=meta.get("target_col", "target_next_return"),
                feature_names=meta.get("feature_names", []),
                metrics=meta.get("metrics", {}),
                metadata=meta,
            )

    def _discover_dl(self) -> None:
        """Discover DL models from models/dl_models."""
        if not DL_MODELS_DIR.exists():
            return
        for sym_dir in DL_MODELS_DIR.iterdir():
            if not sym_dir.is_dir():
                continue
            best_path = sym_dir / "best_dl_model.json"
            if not best_path.exists():
                continue
            try:
                best_doc = json.loads(best_path.read_text())
            except json.JSONDecodeError:
                continue
            model_name = best_doc.get("model_name") or best_doc.get("best_model")
            symbol = best_doc.get("symbol") or sym_dir.name.replace("_NS", ".NS")
            if not model_name:
                continue
            meta_path = sym_dir / f"{model_name}_metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            # DL coexists with ML under a distinct key so /models can list both.
            key = f"{symbol} (dl)"
            self._entries[key] = ModelEntry(
                symbol=symbol, model_name=model_name, model_type="dl", source="dl",
                model_version=meta.get("model_version", "unknown"),
                target_col=meta.get("target_col", "target_next_return"),
                feature_names=meta.get("feature_names", []),
                metrics=meta.get("metrics", {}),
                metadata=meta,
                arch_type=meta.get("arch_type"),
            )

    @staticmethod
    def _read_best(sym_dir: Path, source: str) -> tuple[str, str, dict] | None:
        """Read the deployed model name, symbol, + metadata for one symbol dir.

        The best_*.json files use ``best_model`` (not ``model_name``) as the key
        for the chosen model, and carry the canonical ``symbol`` field — we read
        that rather than guessing from the directory name (which mangles dots).
        Returns (model_name, symbol, metadata) or None if not deployable.
        """
        best_filename = {
            "baseline": "best_model.json",
            "optimized": "best_optimized.json",
        }[source]
        best_path = sym_dir / best_filename
        if not best_path.exists():
            return None
        try:
            best_doc = json.loads(best_path.read_text())
        except json.JSONDecodeError:
            return None
        model_name = best_doc.get("model_name") or best_doc.get("best_model")
        if not model_name:
            return None
        # Prefer the symbol recorded in best_*.json; fall back to dir name.
        symbol = best_doc.get("symbol") or sym_dir.name.replace("_NS", ".NS")
        meta_path = sym_dir / f"{model_name}_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model_name, symbol, meta

    # -------------------------------------------------- access

    def get_entry(self, symbol: str) -> ModelEntry | None:
        """Return the metadata entry for a symbol (None if not deployed)."""
        return self._entries.get(symbol)

    def list_entries(self) -> list[ModelEntry]:
        """All deployed model entries (for GET /models)."""
        return list(self._entries.values())

    def count_symbols(self) -> int:
        """Number of deployed symbols (ML only; DL keys are suffixed)."""
        return sum(1 for e in self._entries.values() if e.model_type == "ml")

    def get_model(self, symbol: str):
        """Lazily load + cache the model object for a symbol.

        Uses the existing prediction-layer loaders (no duplication of load
        logic). Cached after first call so subsequent predictions are fast.
        Raises KeyError if the symbol isn't deployed.
        """
        if symbol in self._loaded:
            return self._loaded[symbol]
        entry = self._entries.get(symbol)
        if entry is None:
            raise KeyError(f"no deployed model for {symbol!r}")
        # Delegate to the verified Phase 7 loaders.
        if entry.model_type == "dl":
            from deep_learning.dl_storage import load_dl_model
            model, _ = load_dl_model(symbol)
        else:
            from models.prediction import load_model
            model, _ = load_model(symbol)
        self._loaded[symbol] = model
        return model

    def is_ready(self) -> bool:
        """True if at least one model is deployed (for the readiness check)."""
        return len(self._entries) > 0
