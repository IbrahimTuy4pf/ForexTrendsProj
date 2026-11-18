"""Causal N-bar momentum: long if return > threshold, short if return < -threshold."""

from __future__ import annotations

import pandas as pd

from quant_forex.config.schema import StrategyConfig
from quant_forex.strategies.common import apply_short_policy


class MomentumStrategy:
    name = "momentum"
    version = "1.0.0"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.params = config.momentum

    def positions(self, frame: pd.DataFrame) -> pd.DataFrame:
        mom = frame["close"].pct_change(periods=self.params.lookback)
        raw = pd.Series(0.0, index=frame.index)
        raw = raw.mask(mom > self.params.threshold, 1.0)
        raw = raw.mask(mom < -self.params.threshold, -1.0)
        raw = raw.where(mom.notna(), 0.0)
        out = frame[["timestamp", "close"]].copy()
        out["position"] = apply_short_policy(raw, self.config.allow_short)
        out["momentum"] = mom
        out["reason"] = [
            _reason(m, p, self.params.lookback, self.params.threshold)
            for m, p in zip(mom.tolist(), out["position"].tolist(), strict=True)
        ]
        return out


def _reason(mom: float, position: float, lookback: int, threshold: float) -> str:
    if pd.isna(mom):
        return f"Warmup: {lookback}-bar momentum not available."
    if position > 0:
        return f"{lookback}-bar return {mom:.6%} above {threshold:.6%}."
    if position < 0:
        return f"{lookback}-bar return {mom:.6%} below -{threshold:.6%}."
    return f"{lookback}-bar return {mom:.6%} inside ±{threshold:.6%} (or shorts disabled)."
