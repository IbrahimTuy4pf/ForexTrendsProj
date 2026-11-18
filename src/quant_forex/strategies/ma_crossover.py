"""Causal SMA crossover: long when fast > slow, short when fast < slow."""

from __future__ import annotations

import pandas as pd

from quant_forex.config.schema import StrategyConfig
from quant_forex.strategies.common import apply_short_policy


class MovingAverageCrossoverStrategy:
    name = "ma_crossover"
    version = "1.0.0"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.params = config.ma_crossover

    def positions(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["close"]
        fast = close.rolling(self.params.fast, min_periods=self.params.fast).mean()
        slow = close.rolling(self.params.slow, min_periods=self.params.slow).mean()
        raw = pd.Series(0.0, index=frame.index)
        raw = raw.mask(fast > slow, 1.0)
        raw = raw.mask(fast < slow, -1.0)
        raw = raw.where(fast.notna() & slow.notna())
        out = frame[["timestamp", "close"]].copy()
        out["position"] = apply_short_policy(raw.fillna(0.0), self.config.allow_short)
        out.loc[fast.isna() | slow.isna(), "position"] = 0.0
        out["fast_sma"] = fast
        out["slow_sma"] = slow
        out["reason"] = [
            _reason(f, s, p)
            for f, s, p in zip(fast.tolist(), slow.tolist(), out["position"].tolist(), strict=True)
        ]
        return out


def _reason(fast: float, slow: float, position: float) -> str:
    if pd.isna(fast) or pd.isna(slow):
        return "Warmup: SMAs not available yet."
    if position > 0:
        return f"Fast SMA {fast:.6f} > slow SMA {slow:.6f}."
    if position < 0:
        return f"Fast SMA {fast:.6f} < slow SMA {slow:.6f}."
    return f"No short (fast {fast:.6f} vs slow {slow:.6f})."
