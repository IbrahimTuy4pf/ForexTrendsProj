"""Donchian-style breakout using the prior N bars only (current bar excluded)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_forex.config.schema import StrategyConfig
from quant_forex.strategies.common import apply_short_policy


class BreakoutStrategy:
    name = "breakout"
    version = "1.0.0"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.params = config.breakout

    def positions(self, frame: pd.DataFrame) -> pd.DataFrame:
        lookback = self.params.lookback
        prior_high = frame["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        prior_low = frame["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        close = frame["close"]
        event = pd.Series(np.nan, index=frame.index, dtype="float64")
        event = event.mask(close > prior_high, 1.0)
        event = event.mask(close < prior_low, -1.0)
        held = event.ffill().fillna(0.0)
        out = frame[["timestamp", "close"]].copy()
        out["position"] = apply_short_policy(held, self.config.allow_short)
        out.loc[prior_high.isna(), "position"] = 0.0
        out["donchian_high"] = prior_high
        out["donchian_low"] = prior_low
        out["reason"] = [
            _reason(c, hi, lo, p)
            for c, hi, lo, p in zip(
                close.tolist(),
                prior_high.tolist(),
                prior_low.tolist(),
                out["position"].tolist(),
                strict=True,
            )
        ]
        return out


def _reason(close: float, high: float, low: float, position: float) -> str:
    if pd.isna(high) or pd.isna(low):
        return "Warmup: prior Donchian channel not available."
    if position > 0:
        return f"Close {close:.6f} vs prior high {high:.6f} (long breakout / hold)."
    if position < 0:
        return f"Close {close:.6f} vs prior low {low:.6f} (short breakout / hold)."
    return f"Inside prior channel [{low:.6f}, {high:.6f}] (or shorts disabled)."
