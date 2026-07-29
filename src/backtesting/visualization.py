"""
StockM v1.0 - Phase 8, Lesson 2 (skeleton)
Visualization
=============

Single Responsibility: render publication-quality charts from a BacktestResult.
Pure rendering over (equity_curve, trades, metrics) -> matplotlib Figures saved
to disk. No computation, no state: it draws what metrics computed.

Why visualization is separate from metrics
------------------------------------------
A metric is a number; a chart is communication. Mixing them couples a number
change to a re-render and makes headless metric computation impossible. Keeping
plots here means Lesson 9 can change the look without touching the numbers, and
the report generator (Lesson 12) just calls these functions and embeds the
files.

Dependency: imports backtesting.trade + matplotlib (imported lazily inside each
function in Lesson 9 to keep the package importable without matplotlib).
Implemented in: Lesson 9.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backtesting.trade import Trade

logger = logging.getLogger("stockm.backtesting.visualization")


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    out_path: Path | None = None,
):
    """Equity curve vs optional buy-and-hold benchmark."""
    raise NotImplementedError("Lesson 9")


def plot_drawdown(equity_curve: pd.Series, out_path: Path | None = None):
    """Drawdown curve (underwater plot)."""
    raise NotImplementedError("Lesson 9")


def plot_monthly_returns(equity_curve: pd.Series, out_path: Path | None = None):
    """Heatmap / bar of returns by calendar month."""
    raise NotImplementedError("Lesson 9")


def plot_trade_distribution(trades: list[Trade], out_path: Path | None = None):
    """Histogram of per-trade P&L."""
    raise NotImplementedError("Lesson 9")


def plot_win_loss_distribution(trades: list[Trade], out_path: Path | None = None):
    """Win / loss distribution by trade."""
    raise NotImplementedError("Lesson 9")
