"""
StockM v1.0 - Phase 8, Lesson 2 (skeleton)
Report Generation
==================

Single Responsibility: assemble a finished BacktestResult into a saved,
human-readable report (summary + metrics + trade statistics + risk analysis +
charts + final recommendation). No computation: it formats what metrics
computed and embeds what visualization rendered.

Why reporting is separate from metrics / visualization
--------------------------------------------------------
The report is the *integration* artifact - it composes the numbers (metrics),
the pictures (visualization), the log (trades), and a recommendation into one
durable document. Keeping it separate means changing the report FORMAT (HTML,
Markdown, PDF) never touches the numbers or the plots. See Lesson 12.

Dependency: imports BacktestResult (from engine), metrics, visualization.
Implemented in: Lesson 12.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backtesting.engine import BacktestResult

logger = logging.getLogger("stockm.backtesting.report")


class ReportGenerator:
    """Assemble + save a full backtesting report from a BacktestResult."""

    def __init__(self, output_dir: Path, config: dict) -> None:
        raise NotImplementedError("Lesson 12")

    def generate(self, result: BacktestResult) -> Path:
        """Write the report in a structured format and return its path."""
        raise NotImplementedError("Lesson 12")
