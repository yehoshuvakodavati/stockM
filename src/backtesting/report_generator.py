"""
StockM v1.0 - Phase 8, Lesson 12
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
Markdown, PDF) never touches the numbers or the plots. The report generator
is a *formatter*, not a calculator.

Format choice: Markdown + JSON sidecar
---------------------------------------
Markdown is the primary report: human-readable in any editor, version-
controllable in git (diffs are meaningful), and renders richly on GitHub/the
IDE. No external dependency (no Jinja2, no PDF engine) - just string assembly.
A JSON sidecar carries the same data machine-readably for Lesson 13's
reproducibility archive and any downstream dashboard. A future HTML/PDF format
is a drop-in replacement here without touching metrics or viz.

The honest mandate (again)
--------------------------
Given ~50% directional accuracy, most strategies lose to Buy & Hold after
costs. The report's "Final Recommendation" must STATE that plainly when it's
the case - never dress up a cost-losing strategy as deployable. A report that
hid the no-edge reality would be the opposite of what the framework is for.

Dependency: imports BacktestResult (from engine), metrics, visualization,
comparison, stress_testing (for the richer report types).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.engine import BacktestResult

logger = logging.getLogger("stockm.backtesting.report")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "backtests" / "reports"


def _md_table(df: pd.DataFrame, floatfmt: str = "+.4f") -> str:
    """Render a DataFrame as a GitHub-Markdown table without the `tabulate` dep.

    The project convention is hand-rolled (no extra packages); pandas'
    ``to_markdown`` requires tabulate, which isn't in the env. This emits the
    same pipe-separated table format directly.
    """
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


class ReportGenerator:
    """Assemble + save a full backtesting report from a BacktestResult.

    Args:
        output_dir: Where to write the report + charts. Default backtests/reports.
        config:     Report config (title, currency, etc.). Minimal in v1.
    """

    def __init__(self, output_dir: Path | str | None = None, config: dict | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_REPORT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    # ------------------------------------------------------------------ API

    def generate(self, result: BacktestResult, render_charts: bool = True) -> Path:
        """Write a single-backtest report (Markdown + JSON) and return its path.

        Args:
            result:         The BacktestResult to report on.
            render_charts:  If True, render the Lesson 9 charts alongside.

        Returns:
            Path to the Markdown report.
        """
        name = result.strategy_name
        chart_paths = self._render_charts(result) if render_charts else {}

        md = self._build_markdown(result, chart_paths)
        md_path = self.output_dir / f"backtest_{name}.md"
        md_path.write_text(md, encoding="utf-8")

        json_path = self.output_dir / f"backtest_{name}.json"
        json_path.write_text(self._build_json(result, chart_paths), encoding="utf-8")

        logger.info("report written -> %s (+ %s)", md_path, json_path)
        return md_path

    def generate_comparison(self, comparison, render_charts: bool = True) -> Path:
        """Write a strategy-comparison report (Lesson 10).

        Args:
            comparison: A backtesting.comparison.ComparisonResult.
            render_charts: If True, render overlaid equity curves.

        Returns:
            Path to the Markdown report.
        """
        chart_paths = {}
        if render_charts and comparison.results:
            from backtesting.visualization import plot_equity_curve, plot_portfolio_growth
            # Overlaid equity curves: each strategy rebased to 1.0.
            growth_series = []
            for res in comparison.results:
                eq = res.equity_curve
                if len(eq) > 0:
                    growth_series.append(eq.astype(float) / float(eq.iloc[0]))
            if growth_series:
                import pandas as pd
                combined = pd.concat(growth_series, axis=1)
                combined.columns = [r.strategy_name for r in comparison.results]
                first = growth_series[0]
                chart_paths["growth"] = plot_portfolio_growth(
                    first, benchmark=None,
                    out_path=self.output_dir / "comparison_growth.png",
                    title="Strategy Comparison (growth of 1.0)",
                )

        md = self._build_comparison_markdown(comparison, chart_paths)
        md_path = self.output_dir / "comparison_report.md"
        md_path.write_text(md, encoding="utf-8")
        json_path = self.output_dir / "comparison_report.json"
        json_path.write_text(self._build_comparison_json(comparison), encoding="utf-8")
        logger.info("comparison report written -> %s", md_path)
        return md_path

    def generate_stress(self, stress, render_charts: bool = True) -> Path:
        """Write a stress-test / regime report (Lesson 11).

        Args:
            stress: A backtesting.stress_testing.StressTestResult.
            render_charts: Unused for now (regime charts are a future add).

        Returns:
            Path to the Markdown report.
        """
        from backtesting.stress_testing import robustness_verdict
        md = self._build_stress_markdown(stress)
        md_path = self.output_dir / "stress_test_report.md"
        md_path.write_text(md, encoding="utf-8")
        verdict = robustness_verdict(stress)
        json_path = self.output_dir / "stress_test_report.json"
        json_path.write_text(json.dumps({
            "regimes": list(stress.by_regime.keys()),
            "strategy_names": stress.strategy_names,
            "summary": stress.summary_table().to_dict(orient="index"),
            "robustness": verdict,
        }, indent=2, default=str), encoding="utf-8")
        logger.info("stress test report written -> %s", md_path)
        return md_path

    # ----------------------------------------------------------- rendering

    def _render_charts(self, result: BacktestResult) -> dict[str, Path]:
        """Render the Lesson 9 chart set for a single backtest."""
        from backtesting import visualization as viz
        paths: dict[str, Path] = {}
        try:
            paths["equity"] = viz.plot_equity_curve(
                result.equity_curve, benchmark=result.benchmark_equity,
                out_path=self.output_dir / f"{result.strategy_name}_equity.png")
            paths["growth"] = viz.plot_portfolio_growth(
                result.equity_curve, benchmark=result.benchmark_equity,
                out_path=self.output_dir / f"{result.strategy_name}_growth.png")
            paths["drawdown"] = viz.plot_drawdown(
                result.equity_curve,
                out_path=self.output_dir / f"{result.strategy_name}_drawdown.png")
            paths["monthly"] = viz.plot_monthly_returns(
                result.equity_curve,
                out_path=self.output_dir / f"{result.strategy_name}_monthly.png")
            paths["trade_dist"] = viz.plot_trade_distribution(
                result.trades,
                out_path=self.output_dir / f"{result.strategy_name}_trade_dist.png")
            paths["win_loss"] = viz.plot_win_loss_distribution(
                result.trades,
                out_path=self.output_dir / f"{result.strategy_name}_win_loss.png")
        except Exception as e:  # noqa: BLE001 - charts are non-fatal
            logger.warning("chart rendering failed: %s", e)
        return paths

    # ----------------------------------------------------------- markdown

    def _build_markdown(self, result: BacktestResult, chart_paths: dict[str, Path]) -> str:
        m = result.metrics
        cur = self.config.get("currency", "₹")
        lines: list[str] = []
        lines.append(f"# Backtest Report: {result.strategy_name}")
        lines.append("")
        lines.append(f"**Symbols:** {', '.join(result.symbols)}  ")
        lines.append(f"**Period:** {result.equity_curve.index[0].date()} → "
                     f"{result.equity_curve.index[-1].date()} "
                     f"({len(result.equity_curve)} bars)  ")
        lines.append(f"**Generated:** {date.today().isoformat()}")
        lines.append("")

        # --- Strategy Summary ---
        lines.append("## Strategy Summary")
        lines.append("")
        lines.append(f"- **Strategy:** `{result.strategy_name}`")
        lines.append(f"- **Initial capital:** {cur}{result.config.get('initial_capital', 0):,.0f}")
        lines.append(f"- **Trades executed:** {int(m['n_trades'])}")
        bm = result.benchmark_equity
        bm_ret = float(bm.iloc[-1] / bm.iloc[0] - 1) if bm is not None and len(bm) else 0.0
        lines.append(f"- **Buy & Hold benchmark:** {bm_ret:+.2%}")
        lines.append(f"- **Beats benchmark:** {'✅ YES' if m['total_return'] > bm_ret else '❌ NO'}")
        lines.append("")

        # --- Performance Metrics ---
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k in ["total_return", "cagr", "sharpe_ratio", "sortino_ratio",
                  "max_drawdown", "calmar_ratio", "profit_factor", "win_rate",
                  "expectancy", "turnover"]:
            v = m[k]
            if k in ("total_return", "cagr", "max_drawdown", "win_rate", "turnover"):
                lines.append(f"| {k} | {v:+.2%} |")
            elif k in ("avg_profit", "avg_loss", "expectancy"):
                lines.append(f"| {k} | {cur}{v:,.2f} |")
            else:
                lines.append(f"| {k} | {v:.4f} |")
        lines.append("")

        # --- Trade Statistics ---
        lines.append("## Trade Statistics")
        lines.append("")
        lines.append(f"- **Total trades:** {int(m['n_trades'])}")
        lines.append(f"- **Win rate:** {m['win_rate']:.2%}")
        lines.append(f"- **Average profit:** {cur}{m['avg_profit']:,.2f}")
        lines.append(f"- **Average loss:** {cur}{m['avg_loss']:,.2f}")
        lines.append(f"- **Expectancy (per trade):** {cur}{m['expectancy']:,.2f}")
        lines.append(f"- **Profit factor:** {m['profit_factor']:.4f}")
        lines.append(f"- **Turnover (annual):** {m['turnover']:.2f}")
        lines.append("")
        # First/last few trades.
        if result.trades:
            lines.append("### Sample Trades (first 10)")
            lines.append("")
            lines.append("| Date | Action | Symbol | Qty | Price | Cost |")
            lines.append("|---|---|---|---|---|---|")
            for t in result.trades[:10]:
                lines.append(f"| {t.date} | {t.action.value} | {t.symbol} | "
                             f"{t.quantity:.4f} | {t.price:.2f} | {t.cost:.2f} |")
            lines.append("")

        # --- Risk Analysis ---
        lines.append("## Risk Analysis")
        lines.append("")
        lines.append(f"- **Maximum drawdown:** {m['max_drawdown']:.2%}")
        lines.append(f"- **Calmar ratio (CAGR/|MDD|):** {m['calmar_ratio']:.4f}")
        lines.append(f"- **Sharpe ratio:** {m['sharpe_ratio']:.4f}")
        lines.append(f"- **Sortino ratio:** {m['sortino_ratio']:.4f}")
        risk_note = self._risk_note(m)
        lines.append(f"- **Assessment:** {risk_note}")
        lines.append("")

        # --- Charts ---
        if chart_paths:
            lines.append("## Charts")
            lines.append("")
            label_map = {"equity": "Equity Curve", "growth": "Portfolio Growth",
                         "drawdown": "Drawdown", "monthly": "Monthly Returns",
                         "trade_dist": "Trade P&L Distribution", "win_loss": "Win/Loss Distribution"}
            for key, p in chart_paths.items():
                rel = p.relative_to(self.output_dir) if p.is_relative_to(self.output_dir) else p
                lines.append(f"### {label_map.get(key, key)}")
                lines.append(f"![{key}]({rel})")
                lines.append("")

        # --- Final Recommendation ---
        lines.append("## Final Recommendation")
        lines.append("")
        lines.append(self._recommendation(result, bm_ret))
        lines.append("")
        return "\n".join(lines)

    def _build_comparison_markdown(self, comparison, chart_paths: dict[str, Path]) -> str:
        lines: list[str] = []
        lines.append("# Strategy Comparison Report")
        lines.append("")
        lines.append(f"**Generated:** {date.today().isoformat()}")
        lines.append(f"**Benchmark:** Buy & Hold ({comparison.benchmark_return:+.2%})")
        lines.append(f"**Recommended:** `{comparison.recommended}` "
                     f"({'beats benchmark ✅' if comparison.recommended_beats_benchmark else 'does NOT beat benchmark ❌'})")
        lines.append("")
        lines.append("## Ranked Summary")
        lines.append("")
        df = comparison.to_dataframe()
        lines.append(_md_table(df))
        lines.append("")
        if chart_paths.get("growth"):
            rel = chart_paths["growth"].relative_to(self.output_dir)
            lines.append("## Equity Growth Comparison")
            lines.append(f"![growth]({rel})")
            lines.append("")
        lines.append("## Verdict")
        lines.append("")
        if comparison.recommended_beats_benchmark:
            lines.append(f"`{comparison.recommended}` beats the Buy & Hold benchmark and is recommended for deployment, "
                         "subject to out-of-sample validation (Lesson 13) and live paper trading.")
        else:
            lines.append(f"**No strategy beats Buy & Hold ({comparison.benchmark_return:+.2%}) after costs.** "
                         f"The honest recommendation is Buy & Hold. `{comparison.recommended}` loses least among the "
                         "candidates, but 'loses least' is not 'wins' - do not deploy a cost-losing strategy.")
        lines.append("")
        return "\n".join(lines)

    def _build_stress_markdown(self, stress) -> str:
        from backtesting.stress_testing import robustness_verdict
        lines: list[str] = []
        lines.append("# Stress Test Report (Regime Analysis)")
        lines.append("")
        lines.append(f"**Generated:** {date.today().isoformat()}")
        lines.append(f"**Regimes tested:** {', '.join(sorted(stress.by_regime.keys()))}")
        lines.append("")
        lines.append("## Returns by Regime")
        lines.append("")
        tbl = stress.summary_table()
        lines.append(_md_table(tbl))
        lines.append("")
        lines.append("## Robustness Verdict")
        lines.append("")
        verdict = robustness_verdict(stress)
        lines.append("| Strategy | Regimes Won | Worst | Best | Sign Stable |")
        lines.append("|---|---|---|---|---|")
        for name, s in verdict.get("strategies", {}).items():
            lines.append(f"| {name} | {s['regimes_won']}/{s['regimes_tested']} | "
                         f"{s['worst_return']:+.2%} | {s['best_return']:+.2%} | "
                         f"{'✅' if s['sign_stable'] else '❌'} |")
        lines.append("")
        lines.append("**Reading:** sign-stable strategies perform consistently across regimes; "
                     "those that flip sign are regime-dependent (fragile). 'Regimes Won' counts "
                     "where the strategy was the top scorer.")
        lines.append("")
        return "\n".join(lines)

    # ----------------------------------------------------------- helpers

    def _risk_note(self, m: dict[str, float]) -> str:
        """A one-line plain-English risk assessment."""
        if m["max_drawdown"] < -0.40:
            dd = "severe drawdown (>40%) - high risk of capitulation"
        elif m["max_drawdown"] < -0.20:
            dd = "moderate drawdown (20-40%)"
        else:
            dd = "shallow drawdown (<20%)"
        if m["sharpe_ratio"] < 0:
            sharpe = "negative risk-adjusted return (volatility not rewarded)"
        elif m["sharpe_ratio"] < 1:
            sharpe = "low Sharpe (<1)"
        else:
            sharpe = f"Sharpe {m['sharpe_ratio']:.2f}"
        return f"{dd}; {sharpe}."

    def _recommendation(self, result: BacktestResult, bm_ret: float) -> str:
        """The honest final recommendation for a single backtest."""
        m = result.metrics
        beats = m["total_return"] > bm_ret
        if not beats:
            return (f"**Do not deploy.** `{result.strategy_name}` returned {m['total_return']:+.2%} "
                    f"vs Buy & Hold {bm_ret:+.2%} - it does not beat the benchmark after costs. "
                    f"The honest action is Buy & Hold. (Expectancy: {m['expectancy']:,.2f}/trade; "
                    f"the model adds no value over holding.)")
        if m["expectancy"] <= 0:
            return (f"**Marginal.** `{result.strategy_name}` beat Buy & Hold on total return but has "
                    f"negative per-trade expectancy ({m['expectancy']:,.2f}) - the edge is not robust. "
                    "Require out-of-sample validation before deploying.")
        return (f"**Candidate for deployment.** `{result.strategy_name}` returned {m['total_return']:+.2%} "
                f"(vs B&H {bm_ret:+.2%}), positive expectancy ({m['expectancy']:,.2f}/trade), "
                f"Sharpe {m['sharpe_ratio']:.2f}. Subject to out-of-sample validation (Lesson 13) "
                "and live paper trading before going live.")

    def _build_json(self, result: BacktestResult, chart_paths: dict[str, Path]) -> str:
        """Machine-readable sidecar for reproducibility (Lesson 13)."""
        return json.dumps({
            "strategy_name": result.strategy_name,
            "symbols": result.symbols,
            "metrics": result.metrics,
            "config": result.config,
            "metadata": result.metadata,
            "n_trades": len(result.trades),
            "charts": {k: str(v) for k, v in chart_paths.items()},
        }, indent=2, default=str)

    def _build_comparison_json(self, comparison) -> str:
        return json.dumps({
            "recommended": comparison.recommended,
            "recommended_beats_benchmark": comparison.recommended_beats_benchmark,
            "benchmark_return": comparison.benchmark_return,
            "rows": [r.__dict__ for r in comparison.rows],
        }, indent=2, default=str)
