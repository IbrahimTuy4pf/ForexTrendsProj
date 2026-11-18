"""Print an indicative size at current equity assumptions (not a live order)."""

from __future__ import annotations

import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.risk.manager import RiskManager, SizingDecision
from quant_forex.strategies.common import true_range_atr
from quant_forex.utils.paths import ProjectPaths


def preview_size(config: AppConfig, paths: ProjectPaths) -> SizingDecision:
    found = latest_processed_csv(paths.data_processed)
    if found is None:
        raise FileNotFoundError("No processed data. Run ingest first.")
    frame = load_processed_ohlcv(found)
    close = float(frame["close"].iloc[-1])
    atr = true_range_atr(frame, config.strategies.atr_window)
    last_atr = atr.iloc[-1]
    stop = None
    if config.strategies.stop_atr_multiple is not None and pd.notna(last_atr):
        stop = close - config.strategies.stop_atr_multiple * float(last_atr)
    vol = (
        frame["close"]
        .pct_change()
        .rolling(config.risk.vol_lookback, min_periods=config.risk.vol_lookback)
        .std()
        .iloc[-1]
    )
    realized = None if pd.isna(vol) else float(vol)
    manager = RiskManager(config.risk, config.backtest.lot_size)
    return manager.size(
        equity=config.backtest.initial_capital,
        entry=close,
        stop=stop,
        realized_vol=realized,
    )
