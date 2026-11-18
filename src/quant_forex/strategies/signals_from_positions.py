"""Turn a position series into TradingSignal objects (entry on position change)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_forex.strategies.common import stops_from_atr, true_range_atr
from quant_forex.strategies.signal import Direction, TradingSignal


def position_to_direction(value: float) -> Direction:
    if value > 0:
        return "LONG"
    if value < 0:
        return "SHORT"
    return "FLAT"


def build_entry_signals(
    ohlcv: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    instrument: str,
    strategy_name: str,
    strategy_version: str,
    atr_window: int,
    stop_atr_multiple: float | None,
    take_profit_atr_multiple: float | None,
    risk_percentage: float | None,
) -> list[TradingSignal]:
    aligned = ohlcv.copy().reset_index(drop=True)
    pos = positions["position"].fillna(0.0).reset_index(drop=True)
    changed = pos.ne(pos.shift(1, fill_value=0.0)) & (pos != 0.0)
    atr = true_range_atr(aligned, atr_window)
    stop, take = stops_from_atr(
        pos, aligned["close"], atr, stop_atr_multiple, take_profit_atr_multiple
    )
    signals: list[TradingSignal] = []
    for idx in aligned.index[changed.to_numpy()]:
        direction = position_to_direction(float(pos.iloc[idx]))
        if direction == "FLAT":
            continue
        extra: dict[str, Any] = {}
        for col in positions.columns:
            if col in {"timestamp", "close", "position", "reason"}:
                continue
            value = positions.iloc[idx][col]
            extra[col] = None if pd.isna(value) else _jsonish(value)
        ts = aligned.iloc[idx]["timestamp"]
        if hasattr(ts, "isoformat"):
            ts_text = ts.isoformat()
        else:
            ts_text = str(ts)
        signals.append(
            TradingSignal(
                timestamp=ts_text,
                instrument=instrument,
                direction=direction,
                probability=None,
                entry=float(aligned.iloc[idx]["close"]),
                stop_loss=_optional(stop.iloc[idx]),
                take_profit=_optional(take.iloc[idx]),
                position_size=None,
                risk_percentage=risk_percentage,
                model_version=None,
                strategy_name=strategy_name,
                strategy_version=strategy_version,
                reason=str(positions.iloc[idx]["reason"]),
                features=extra,
            )
        )
    return signals


def _optional(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _jsonish(value: object) -> object:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    return value
