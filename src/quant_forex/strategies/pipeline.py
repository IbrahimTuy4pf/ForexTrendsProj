"""Run enabled baseline strategies and write signals plus naive diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_forex.config.schema import AppConfig, StrategyConfig
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.strategies.breakout import BreakoutStrategy
from quant_forex.strategies.buy_and_hold import BuyAndHoldStrategy
from quant_forex.strategies.diagnostics import NaiveDiagnostics, naive_next_bar_diagnostics
from quant_forex.strategies.ma_crossover import MovingAverageCrossoverStrategy
from quant_forex.strategies.mean_reversion import MeanReversionStrategy
from quant_forex.strategies.momentum import MomentumStrategy
from quant_forex.strategies.signal import TradingSignal
from quant_forex.strategies.signals_from_positions import build_entry_signals
from quant_forex.utils.paths import ProjectPaths


@dataclass
class StrategyRun:
    name: str
    positions: pd.DataFrame
    signals: list[TradingSignal]
    diagnostics: NaiveDiagnostics


@dataclass
class BaselinePipelineResult:
    input_path: Path
    positions_path: Path
    signals_path: Path
    report_path: Path
    json_path: Path
    runs: list[StrategyRun]


def enabled_strategies(config: StrategyConfig) -> list[object]:
    mapping: list[tuple[bool, object]] = [
        (config.buy_and_hold.enabled, BuyAndHoldStrategy(config)),
        (config.ma_crossover.enabled, MovingAverageCrossoverStrategy(config)),
        (config.momentum.enabled, MomentumStrategy(config)),
        (config.mean_reversion.enabled, MeanReversionStrategy(config)),
        (config.breakout.enabled, BreakoutStrategy(config)),
    ]
    return [strategy for enabled, strategy in mapping if enabled]


def run_baselines(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
) -> BaselinePipelineResult:
    if input_csv is None:
        found = latest_processed_csv(paths.data_processed)
        if found is None:
            raise FileNotFoundError("No *_processed.csv found. Run ingest first.")
        input_path = found
    else:
        candidate = Path(input_csv)
        input_path = candidate if candidate.is_file() else paths.resolve(candidate)

    ohlcv = load_processed_ohlcv(input_path)
    symbol = (
        str(ohlcv["symbol"].iloc[0])
        if "symbol" in ohlcv.columns
        else config.instrument.symbol
    )
    runs: list[StrategyRun] = []
    position_frames: list[pd.DataFrame] = []
    signal_rows: list[dict[str, object]] = []

    for strategy in enabled_strategies(config.strategies):
        pos = strategy.positions(ohlcv)
        signals = build_entry_signals(
            ohlcv,
            pos,
            instrument=symbol,
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            atr_window=config.strategies.atr_window,
            stop_atr_multiple=config.strategies.stop_atr_multiple,
            take_profit_atr_multiple=config.strategies.take_profit_atr_multiple,
            risk_percentage=config.risk.risk_per_trade,
        )
        diagnostics = naive_next_bar_diagnostics(
            ohlcv, pos["position"], strategy.name, n_entries=len(signals)
        )
        tagged = pos.copy()
        tagged.insert(0, "strategy", strategy.name)
        tagged.insert(1, "instrument", symbol)
        position_frames.append(tagged)
        signal_rows.extend(sig.to_row() for sig in signals)
        runs.append(
            StrategyRun(
                name=strategy.name,
                positions=pos,
                signals=signals,
                diagnostics=diagnostics,
            )
        )

    stem = input_path.stem.replace("_processed", "")
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    positions_path = paths.data_processed / f"{stem}_baseline_positions.csv"
    signals_path = paths.data_processed / f"{stem}_baseline_signals.csv"
    pd.concat(position_frames, ignore_index=True).to_csv(positions_path, index=False)
    pd.DataFrame(signal_rows).to_csv(signals_path, index=False)

    report_path = paths.reports / f"{stem}_baselines.md"
    json_path = paths.reports / f"{stem}_baselines.json"
    report_path.write_text(_markdown(runs, input_path, config), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "instrument": symbol,
                "allow_short": config.strategies.allow_short,
                "diagnostics": [item.diagnostics.model_dump() for item in runs],
                "n_signals": {item.name: len(item.signals) for item in runs},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return BaselinePipelineResult(
        input_path=input_path,
        positions_path=positions_path,
        signals_path=signals_path,
        report_path=report_path,
        json_path=json_path,
        runs=runs,
    )


def _markdown(runs: list[StrategyRun], input_path: Path, config: AppConfig) -> str:
    rows = []
    for item in runs:
        d = item.diagnostics
        hit = f"{d.naive_hit_rate:.3f}" if d.naive_hit_rate is not None else "n/a"
        total = (
            f"{d.naive_sum_next_return:.6f}" if d.naive_sum_next_return is not None else "n/a"
        )
        rows.append(
            f"| `{d.strategy}` | {d.n_long} | {d.n_short} | {d.n_flat} | {d.n_entries} | "
            f"{hit} | {total} |"
        )
    examples = []
    for item in runs:
        if not item.signals:
            continue
        sig = item.signals[0]
        examples.append(
            f"- **{sig.strategy_name}** `{sig.direction}` @ {sig.timestamp}: entry={sig.entry}, "
            f"stop={sig.stop_loss}, take={sig.take_profit}, risk={sig.risk_percentage}, "
            f"reason={sig.reason}"
        )
    return f"""# Baseline strategies (Phase 4)

Input: `{input_path.as_posix()}`  
Instrument: `{config.instrument.symbol}`  
Allow short: `{config.strategies.allow_short}`  
Costs in this report: **not applied** (spread={config.costs.spread_pips} pips, slippage={config.costs.slippage_pips} pips are recorded for Phase 5).

This is **not** a backtest. Naive figures multiply the position at bar *t* by the **next** bar’s simple return. No spread, slippage, swaps, or stop/target fills.

Later ML models must beat these rules **after** realistic costs and risk limits, not on this table alone.

## Naive next-bar comparison

| Strategy | Long bars | Short bars | Flat | Entries | Hit rate | Sum of next-bar strategy returns |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Hit rate is the share of active next-bar returns that were positive. On noise, values near 0.5 are expected.

## Example signals (first entry per strategy)

{chr(10).join(examples) if examples else "_No directional entries._"}

Position size is left empty until the risk module (Phase 6). Stop/take use ATR multiples from config when ATR is defined.
"""
