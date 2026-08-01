"""
StockM v1.0 - Phase 8, Lesson 9
Visualization
=============

Single Responsibility: render publication-quality charts from an equity curve
and trade log. PURE RENDERING over (equity_curve, trades) -> matplotlib Figures
saved to disk. No metric computation, no state: it draws what the portfolio
recorded and what metrics computed. A chart is communication, not calculation.

Why visualization is separate from metrics
------------------------------------------
A metric is a number; a chart is communication. Mixing them couples a number
change to a re-render and makes headless metric computation impossible. Keeping
plots here means Lesson 9 can change the look without touching the numbers, and
the report generator (Lesson 12) just calls these functions and embeds the
files.

Convention (mirrors deep_learning/plotting.py)
----------------------------------------------
- matplotlib imported lazily inside each function so the backtesting package
  stays importable on a machine without matplotlib (e.g. a pure-metric CI run).
- Headless backend ("Agg"): no display, writes PNG to disk.
- dpi=120, tight_layout, plt.close(fig) after every save (no figure leaks).
- Every function returns the Path it wrote (so the report can embed it).
- Default output dir: PROJECT_ROOT / "reports" / "backtesting".

Charts
------
    plot_equity_curve          - equity vs optional buy-and-hold benchmark
    plot_portfolio_growth      - equity rebased to 1.0 (growth of ₹1)
    plot_drawdown              - underwater plot (drawdown over time)
    plot_monthly_returns       - calendar heatmap of monthly returns
    plot_trade_distribution    - histogram of per-trade P&L
    plot_win_loss_distribution - side-by-side win/loss P&L boxes + means
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.trade import Trade, TradeAction

logger = logging.getLogger("stockm.backtesting.visualization")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BT_REPORTS_DIR = PROJECT_ROOT / "reports" / "backtesting"


def _resolve_out_path(out_path: Path | str | None, default_name: str) -> Path:
    """Normalise an output path; default to reports/backtesting/<name>."""
    p = Path(out_path) if out_path else (BT_REPORTS_DIR / default_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _import_mpl():
    """Lazy matplotlib import + headless backend (keeps package importable)."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display, write PNG
    import matplotlib.pyplot as plt

    return plt


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    out_path: Path | str | None = None,
    title: str = "Equity Curve",
) -> Path:
    """Equity curve in currency units, with an optional buy-and-hold benchmark.

    Args:
        equity_curve: Date-indexed total portfolio value.
        benchmark:    Optional date-indexed benchmark equity (e.g. Buy & Hold),
                      reindexed to the same start capital for fair comparison.
        out_path:     PNG destination. None -> reports/backtesting/equity_curve.png.
        title:        Plot title.

    Returns:
        The path written.
    """
    if equity_curve is None or len(equity_curve) == 0:
        raise ValueError("empty equity curve - nothing to plot")
    plt = _import_mpl()
    out = _resolve_out_path(out_path, "equity_curve.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve.index, equity_curve.values, color="#1f77b4",
            linewidth=1.5, label="strategy")
    if benchmark is not None and len(benchmark) > 0:
        bm = benchmark.reindex(equity_curve.index).ffill()
        ax.plot(bm.index, bm.values, color="#d62728", linewidth=1.2,
                linestyle="--", alpha=0.8, label="buy & hold")
        ax.legend(loc="upper left")
    ax.set_xlabel("date")
    ax.set_ylabel("portfolio value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Currency formatting on the y-axis.
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("equity curve saved -> %s", out)
    return out


def plot_portfolio_growth(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    out_path: Path | str | None = None,
    title: str = "Portfolio Growth (rebased to 1.0)",
) -> Path:
    """Equity rebased to 1.0 - "growth of ₹1 invested".

    Normalises both strategy and benchmark to start at 1.0 so the chart shows
    pure multiplicative growth, independent of capital size. A value of 1.15 =
    +15%; 0.90 = -10%. This is the form used in finance papers because it makes
    strategies of different capital sizes directly comparable.
    """
    if equity_curve is None or len(equity_curve) == 0:
        raise ValueError("empty equity curve - nothing to plot")
    plt = _import_mpl()
    out = _resolve_out_path(out_path, "portfolio_growth.png")

    strat = equity_curve.astype(float) / float(equity_curve.iloc[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strat.index, strat.values, color="#1f77b4", linewidth=1.5, label="strategy")
    if benchmark is not None and len(benchmark) > 0:
        bm = benchmark.reindex(equity_curve.index).ffill().astype(float)
        bm = bm / float(bm.iloc[0])
        ax.plot(bm.index, bm.values, color="#d62728", linewidth=1.2,
                linestyle="--", alpha=0.8, label="buy & hold")
        ax.legend(loc="upper left")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)  # break-even line
    ax.set_xlabel("date")
    ax.set_ylabel("growth of 1.0")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("portfolio growth saved -> %s", out)
    return out


def plot_drawdown(
    equity_curve: pd.Series, out_path: Path | str | None = None,
    title: str = "Drawdown (Underwater Plot)",
) -> Path:
    """Drawdown over time - the "underwater" chart.

    Plots the % below the running peak at each bar. The deeper the curve dips,
    the more the portfolio was underwater at that point. Filled below zero so
    the pain is visually obvious. This is the chart a risk manager stares at
    most: a strategy that spends long periods far underwater is hard to hold.
    """
    if equity_curve is None or len(equity_curve) < 2:
        raise ValueError("need >= 2 points to compute drawdown")
    plt = _import_mpl()
    out = _resolve_out_path(out_path, "drawdown.png")

    curve = equity_curve.astype(float)
    running_max = curve.cummax()
    drawdowns = (curve - running_max) / running_max  # <= 0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdowns.index, drawdowns.values, 0, color="#d62728", alpha=0.4)
    ax.plot(drawdowns.index, drawdowns.values, color="#d62728", linewidth=1)
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("drawdown plot saved -> %s", out)
    return out


def plot_monthly_returns(
    equity_curve: pd.Series, out_path: Path | str | None = None,
    title: str = "Monthly Returns Heatmap",
) -> Path:
    """Calendar heatmap of monthly returns (year x month).

    Computes the return for each calendar month from the equity curve and
    renders a year-rows x month-columns heatmap. Green = positive month, red =
    negative. Reveals seasonality and consistency: a strategy that's green most
    months with occasional red is robust; one with scattered large red months
    is fragile. NaN months (no data) are greyed out.
    """
    if equity_curve is None or len(equity_curve) < 2:
        raise ValueError("need >= 2 points for monthly returns")
    plt = _import_mpl()
    out = _resolve_out_path(out_path, "monthly_returns.png")

    # Resample to month-end; monthly return = pct change of month-end equity.
    monthly = equity_curve.resample("ME").last().pct_change().dropna()
    if len(monthly) == 0:
        raise ValueError("no full months in equity curve")
    # Pivot to year (rows) x month (cols).
    frame = pd.DataFrame({"ret": monthly})
    frame["year"] = monthly.index.year
    frame["month"] = monthly.index.month
    pivot = frame.pivot_table(index="year", columns="month", values="ret")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, max(3, 0.6 * len(pivot) + 1)))
    # Diverging colormap centred at 0: red (neg) <-> green (pos).
    vmax = float(np.nanmax(np.abs(pivot.values))) or 0.05
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    # Annotate each cell with the return %.
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="monthly return")
    ax.set_title(title)
    ax.set_xlabel("month")
    ax.set_ylabel("year")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("monthly returns heatmap saved -> %s", out)
    return out


def _round_trip_pnls(trades: list[Trade]) -> list[float]:
    """FIFO per-round-trip realised P&L (mirrors metrics._trade_pnl)."""
    buy_lots: list[list[float]] = []  # [price, qty_remaining, cost]
    pnls: list[float] = []
    for t in trades:
        if t.action == TradeAction.BUY:
            buy_lots.append([t.price, t.quantity, t.cost])
        elif t.action == TradeAction.SELL:
            remaining = t.quantity
            proceeds_net = (t.price * t.quantity) - t.cost
            lot_proceeds = 0.0
            lot_costs = 0.0
            while remaining > 1e-12 and buy_lots:
                lot = buy_lots[0]
                take = min(lot[1], remaining)
                frac = take / t.quantity if t.quantity > 0 else 0
                lot_proceeds += frac * proceeds_net
                lot_costs += (lot[0] * take) + (frac * lot[2])
                lot[1] -= take
                remaining -= take
                if lot[1] <= 1e-12:
                    buy_lots.pop(0)
            if t.quantity > 0:
                pnls.append(lot_proceeds - lot_costs)
    return pnls


def plot_trade_distribution(
    trades: list[Trade], out_path: Path | str | None = None,
    title: str = "Trade P&L Distribution",
) -> Path:
    """Histogram of per-round-trip realised P&L.

    Shows the shape of trade outcomes: a right-skewed distribution (a fat left
    tail of large losses) is the signature of a strategy that occasionally
    blows up; a symmetric distribution around a positive mean is healthy. The
    zero line is marked so wins/losses are visually distinct.
    """
    plt = _import_mpl()
    out = _resolve_out_path(out_path, "trade_distribution.png")
    pnls = _round_trip_pnls(trades)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not pnls:
        ax.text(0.5, 0.5, "no round-trip trades", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    else:
        colors = ["#2ca02c" if p >= 0 else "#d62728" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, width=0.8, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("trade #")
        ax.set_ylabel("realised P&L")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("trade distribution saved -> %s", out)
    return out


def plot_win_loss_distribution(
    trades: list[Trade], out_path: Path | str | None = None,
    title: str = "Win / Loss Distribution",
) -> Path:
    """Side-by-side box plot of winning vs losing trade P&L.

    Two boxes: wins (green) and losses (red), showing median, quartiles, and
    outliers. Reveals the *size* asymmetry that win_rate hides: if the loss box
    extends far below the win box, the strategy loses more when it loses than it
    makes when it wins - the "win small, lose big" pathology. Means are marked
    so expectancy is visible at a glance.
    """
    plt = _import_mpl()
    out = _resolve_out_path(out_path, "win_loss_distribution.png")
    pnls = _round_trip_pnls(trades)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    if not wins and not losses:
        ax.text(0.5, 0.5, "no round-trip trades", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    else:
        data = [wins if wins else [0], losses if losses else [0]]
        positions = [1, 2]
        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markeredgecolor="black", markersize=6))
        for patch, color in zip(bp["boxes"], ["#2ca02c", "#d62728"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"wins\n(n={len(wins)})", f"losses\n(n={len(losses)})"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("realised P&L")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("win/loss distribution saved -> %s", out)
    return out
