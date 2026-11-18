"""Markdown exploration report."""

from __future__ import annotations

from pathlib import Path

from quant_forex.exploration.summary import ExplorationSummary


def exploration_markdown(
    summary: ExplorationSummary,
    figure_paths: list[Path],
    relative_to: Path | None = None,
) -> str:
    weekday_lines = [
        f"| {day} | {value:.6f} |" if value is not None else f"| {day} | n/a |"
        for day, value in summary.weekday_mean_return.items()
    ] or ["| (none) | |"]
    ac_lines = [
        f"| {lag} | {value:.4f} |" if value is not None else f"| {lag} | n/a |"
        for lag, value in summary.return_autocorr.items()
    ] or ["| (none) | |"]
    fig_lines = []
    for path in figure_paths:
        display = path.name
        if relative_to is not None:
            try:
                display = path.relative_to(relative_to).as_posix()
            except ValueError:
                display = str(path)
        if path.suffix.lower() == ".png":
            fig_lines.append(f"![{path.stem}]({display})")
        else:
            fig_lines.append(f"- [{path.name}]({display})")
    caveats = "\n".join(f"- {item}" for item in summary.caveats)
    jb_p = (
        f"{summary.jarque_bera_pvalue:.4g}"
        if summary.jarque_bera_pvalue is not None
        else "n/a"
    )
    return f"""# Exploration report (Phase 2)

Symbol: `{summary.symbol}`  
Timeframe: `{summary.timeframe}`  
Rows: {summary.n_rows}  
Range: {summary.date_start} → {summary.date_end}

This report **describes** the series. It is not a trading strategy, not a backtest, and not evidence of profitability.

## Price

| Field | Value |
| --- | --- |
| Close min | {summary.close_min} |
| Close max | {summary.close_max} |
| Missing expected bars | {summary.n_missing_expected_bars} |
| Longest gap (bars) | {summary.longest_gap_bars} |
| Sessions applicable | {summary.sessions_applicable} |

## Returns (close-to-close)

Uses only the previous close. First row is undefined (NaN) by construction.

| Field | Value |
| --- | --- |
| Mean | {summary.simple_return_mean} |
| Std (sample) | {summary.simple_return_std} |
| Skew | {summary.simple_return_skew} |
| Excess kurtosis | {summary.simple_return_excess_kurtosis} |
| Jarque–Bera statistic | {summary.jarque_bera_statistic} |
| Jarque–Bera p-value | {jb_p} |
| Lag-1 autocorrelation | {summary.lag1_return_autocorr} |

### Autocorrelation

| Lag | corr |
| --- | --- |
{chr(10).join(ac_lines)}

### Mean return by weekday

These are **unconditional averages**, not a day-of-week trading rule.

| Weekday | Mean simple return |
| --- | --- |
{chr(10).join(weekday_lines)}

## Figures

{chr(10).join(fig_lines) if fig_lines else "_No figures written._"}

## Caveats

{caveats}
"""
