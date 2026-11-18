"""Run costed backtests for enabled baseline strategies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.backtesting.engine import BacktestResult, run_backtest
from quant_forex.backtesting.metrics import compute_metrics, trade_frame
from quant_forex.backtesting.report import backtest_markdown, save_equity_charts
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.strategies.common import stops_from_atr, true_range_atr
from quant_forex.strategies.pipeline import enabled_strategies
from quant_forex.utils.paths import ProjectPaths


@dataclass
class BacktestPipelineResult:
    input_path: Path
    report_path: Path
    json_path: Path
    trades_path: Path
    equity_path: Path
    figure_dir: Path
    results: list[BacktestResult]


def run_backtest_pipeline(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
    strategy_name: str | None = None,
) -> BacktestPipelineResult:
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
    atr = true_range_atr(ohlcv, config.strategies.atr_window)
    realized_vol = ohlcv["close"].pct_change().rolling(
        config.risk.vol_lookback, min_periods=config.risk.vol_lookback
    ).std()
    strategies = enabled_strategies(config.strategies)
    if strategy_name:
        strategies = [item for item in strategies if item.name == strategy_name]
        if not strategies:
            raise ValueError(f"Unknown or disabled strategy {strategy_name!r}.")

    results: list[BacktestResult] = []
    metric_rows: list[tuple[BacktestResult, dict[str, object]]] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    stem = input_path.stem.replace("_processed", "")
    figure_dir = paths.reports / "figures" / f"{stem}_backtest"

    for strategy in strategies:
        pos = strategy.positions(ohlcv)
        stop, take = stops_from_atr(
            pos["position"],
            ohlcv["close"],
            atr,
            config.strategies.stop_atr_multiple,
            config.strategies.take_profit_atr_multiple,
        )
        result = run_backtest(
            ohlcv,
            pos["position"],
            stop=stop,
            take=take,
            config=config,
            strategy_name=strategy.name,
            realized_vol=realized_vol,
        )
        results.append(result)
        metric_rows.append((result, compute_metrics(result)))
        tf = trade_frame(result.trades)
        if not tf.empty:
            trade_frames.append(tf)
        eq = result.equity_curve.copy()
        eq.insert(0, "strategy", strategy.name)
        equity_frames.append(eq)
        save_equity_charts(result, figure_dir)

    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    trades_path = paths.data_processed / f"{stem}_backtest_trades.csv"
    equity_path = paths.data_processed / f"{stem}_backtest_equity.csv"
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(trades_path, index=False)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)
    pd.concat(equity_frames, ignore_index=True).to_csv(equity_path, index=False)

    report_path = paths.reports / f"{stem}_backtest.md"
    json_path = paths.reports / f"{stem}_backtest.json"
    report_path.write_text(backtest_markdown(metric_rows, instrument=symbol), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "instrument": symbol,
                "metrics": [
                    {"strategy": result.strategy, **{k: _json(v) for k, v in metrics.items()}}
                    for result, metrics in metric_rows
                ],
                "halt": {result.strategy: result.halt_reason for result in results},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return BacktestPipelineResult(
        input_path=input_path,
        report_path=report_path,
        json_path=json_path,
        trades_path=trades_path,
        equity_path=equity_path,
        figure_dir=figure_dir,
        results=results,
    )


def _json(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float):
        return None if pd.isna(value) else value
    return value
