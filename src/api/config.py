"""
StockM v1.0 - Phase 9, Lesson 2 (minimal)
API Configuration
=================

A minimal settings object for Lesson 2. Lesson 3 expands this into a full
env-driven settings manager (pydantic-settings) supporting dev/test/prod
profiles, logging config, and API versioning. For now we expose the few values
``main.py`` needs to build the app.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Settings:
    """Frozen settings (immutable -> safe to share across requests/threads).

    Frozen so no handler can accidentally mutate a global setting mid-flight.
    Lesson 3 replaces this with pydantic BaseSettings for env binding.
    """

    api_title: str = "StockM Prediction API"
    api_version: str = "1.0.0"
    # Default to the project's existing prepared test split (unseen data).
    default_split: str = "test"
    default_threshold: float = 0.0


settings = _Settings()
