"""Markdown + chart output for backtests."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quant_forex.backtesting.engine import BacktestResult


def save_equity_charts(result: BacktestResult, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    eq_path = directory / f"{result.strategy}_equity.png"
    dd_path = directory / f"{result.strategy}_drawdown.png"
    curve = result.equity_curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve["timestamp"], curve["equity"], color="#1f4e79")
    ax.set_title(f"{result.strategy} equity (research backtest, costs applied)")
    ax.set_ylabel("account equity")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(eq_path, dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve["timestamp"], curve["drawdown"], color="#9b2226")
    ax.set_title(f"{result.strategy} drawdown fraction")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(dd_path, dpi=120)
    plt.close(fig)
    return eq_path, dd_path


def backtest_markdown(
    rows: list[tuple[BacktestResult, dict[str, object]]],
    *,
    instrument: str,
) -> str:
    lines = [
        f"| `{result.strategy}` | {metrics.get('number_of_trades')} | "
        f"{_fmt(metrics.get('total_return'))} | {_fmt(metrics.get('max_drawdown'))} | "
        f"{_fmt(metrics.get('sharpe_ratio'))} | {_fmt(metrics.get('win_rate'))} | "
        f"{_fmt(metrics.get('profit_factor'))} | {result.halted} |"
        for result, metrics in rows
    ]
    assumptions = rows[0][0].assumptions if rows else {}
    return f"""# Backtest report (Phase 5)

Instrument: `{instrument}`

Fills are **next bar open** after a close-based signal. Spread and slippage are applied. If stop and take-profit both trade in the same bar, **stop is assumed** (worst case). This is still a model of OHLC bars, not tick data, and **not** a live result.

Assumptions: {assumptions}

| Strategy | Trades | Total return | Max DD | Sharpe | Win rate | Profit factor | Halted |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(lines)}

Do not treat a positive total return as an edge. Run `python -m quant_forex diagnose` before treating any number as research evidence.
"""


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
