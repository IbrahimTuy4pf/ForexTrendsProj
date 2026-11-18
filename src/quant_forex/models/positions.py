"""Map classifier probabilities to positions and optional costed overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_forex.backtesting.engine import run_backtest
from quant_forex.backtesting.metrics import compute_metrics
from quant_forex.config.schema import AppConfig
from quant_forex.strategies.common import stops_from_atr, true_range_atr


def proba_to_position(proba: np.ndarray, long_th: float, short_th: float) -> np.ndarray:
    out = np.zeros(len(proba), dtype=float)
    out[proba >= long_th] = 1.0
    out[proba <= short_th] = -1.0
    return out


def costed_overlay_backtest(
    ohlcv: pd.DataFrame,
    signal_rows: pd.DataFrame,
    positions: np.ndarray,
    config: AppConfig,
    strategy_name: str,
    note: str,
) -> dict[str, Any] | None:
    bars = ohlcv.sort_values("timestamp").reset_index(drop=True)
    lookup = pd.Series(
        positions,
        index=pd.to_datetime(signal_rows["timestamp"], utc=True),
    )
    lookup = lookup[~lookup.index.duplicated(keep="last")]
    pos = (
        pd.to_datetime(bars["timestamp"], utc=True)
        .map(lookup)
        .fillna(0.0)
        .astype(float)
    )
    atr = true_range_atr(bars, config.strategies.atr_window)
    stop, take = stops_from_atr(
        pos,
        bars["close"],
        atr,
        config.strategies.stop_atr_multiple,
        config.strategies.take_profit_atr_multiple,
    )
    result = run_backtest(
        bars,
        pos,
        stop=stop,
        take=take,
        config=config,
        strategy_name=strategy_name,
        realized_vol=bars["close"]
        .pct_change()
        .rolling(config.risk.vol_lookback, min_periods=config.risk.vol_lookback)
        .std(),
    )
    metrics = compute_metrics(result)
    return {
        "note": note,
        "number_of_trades": metrics.get("number_of_trades"),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
    }
