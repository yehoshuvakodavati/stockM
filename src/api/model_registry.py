"""
StockM v1.0 - Phase 9, Lesson 2 (stub)
Model Registry — minimal version
=================================

A placeholder registry so the API composition root (main.py) and dependency
injection (dependencies.py) resolve and the server can start in Lesson 2.
Lesson 4 replaces this with the full registry: multiple versions, ML+DL,
metadata, automatic discovery + loading from models/saved_models and
models/optimized.

Lesson 2 contract (what main.py calls):
    registry = ModelRegistry()
    registry.load_all()          # discover + load deployed models
    registry.count_symbols()     # how many symbols have a loaded model
"""
from __future__ import annotations

import logging

logger = logging.getLogger("stockm.api.registry")


class ModelRegistry:
    """Minimal registry: counts deployed-model directories without loading.

    Lesson 4 will load the actual model objects + metadata. For Lesson 2 we
    only verify the wiring: the lifespan calls ``load_all`` and reports a count.
    """

    def __init__(self) -> None:
        self._symbols: set[str] = set()

    def load_all(self) -> None:
        """Discover deployed models (Lesson 4 loads them; here we just count)."""
        from pathlib import Path

        from models.model_storage import SAVED_MODELS_DIR

        # Count symbols that have a saved-model directory (a deployed model).
        if SAVED_MODELS_DIR.exists():
            self._symbols = {
                p.name for p in SAVED_MODELS_DIR.iterdir() if p.is_dir()
            }
        logger.info("registry discovered %d deployed symbols", len(self._symbols))

    def count_symbols(self) -> int:
        """Number of symbols with a deployed model."""
        return len(self._symbols)
