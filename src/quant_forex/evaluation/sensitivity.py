"""Small parameter grids around configured rules. Not an optimizer."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_forex.backtesting.engine import run_backtest
from quant_forex.backtesting.metrics import compute_metrics
from quant_forex.config.schema import AppConfig
from quant_forex.strategies.common import stops_from_atr, true_range_atr
from quant_forex.strategies.ma_crossover import MovingAverageCrossoverStrategy
from quant_forex.strategies.momentum import MomentumStrategy


def run_costed_strategy(ohlcv: pd.DataFrame, config: AppConfig, strategy: Any):
    atr = true_range_atr(ohlcv, config.strategies.atr_window)
    realized_vol = (
        ohlcv["close"]
        .pct_change()
        .rolling(config.risk.vol_lookback, min_periods=config.risk.vol_lookback)
        .std()
    )
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
    return result, compute_metrics(result), pos["position"]


def ma_crossover_grid(ohlcv: pd.DataFrame, config: AppConfig) -> list[dict[str, Any]]:
    base_fast = config.strategies.ma_crossover.fast
    base_slow = config.strategies.ma_crossover.slow
    rows: list[dict[str, Any]] = []
    for dfast in config.diagnostics.ma_fast_offsets:
        for dslow in config.diagnostics.ma_slow_offsets:
            fast = base_fast + dfast
            slow = base_slow + dslow
            if fast < 2 or slow < 3 or fast >= slow:
                continue
            variant = _with_ma(config, fast, slow)
            result, metrics, _ = run_costed_strategy(
                ohlcv, variant, MovingAverageCrossoverStrategy(variant.strategies)
            )
            rows.append(
                {
                    "fast": fast,
                    "slow": slow,
                    "is_configured": fast == base_fast and slow == base_slow,
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "total_return": metrics.get("total_return"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "number_of_trades": metrics.get("number_of_trades"),
                    "halted": result.halted,
                }
            )
    return rows


def momentum_grid(ohlcv: pd.DataFrame, config: AppConfig) -> list[dict[str, Any]]:
    base = config.strategies.momentum.lookback
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for delta in config.diagnostics.momentum_lookback_offsets:
        lookback = base + delta
        if lookback < 2 or lookback in seen:
            continue
        seen.add(lookback)
        variant = _with_momentum_lookback(config, lookback)
        result, metrics, _ = run_costed_strategy(
            ohlcv, variant, MomentumStrategy(variant.strategies)
        )
        rows.append(
            {
                "lookback": lookback,
                "is_configured": lookback == base,
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "total_return": metrics.get("total_return"),
                "max_drawdown": metrics.get("max_drawdown"),
                "number_of_trades": metrics.get("number_of_trades"),
                "halted": result.halted,
            }
        )
    return rows


def _with_ma(config: AppConfig, fast: int, slow: int) -> AppConfig:
    payload = config.model_dump()
    payload["strategies"]["ma_crossover"]["fast"] = fast
    payload["strategies"]["ma_crossover"]["slow"] = slow
    return AppConfig.model_validate(payload)


def _with_momentum_lookback(config: AppConfig, lookback: int) -> AppConfig:
    payload = config.model_dump()
    payload["strategies"]["momentum"]["lookback"] = lookback
    return AppConfig.model_validate(payload)
