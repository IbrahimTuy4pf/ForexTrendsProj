"""Causal z-score mean reversion vs a rolling mean (not a full-sample mean)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_forex.config.schema import StrategyConfig
from quant_forex.strategies.common import apply_short_policy


class MeanReversionStrategy:
    name = "mean_reversion"
    version = "1.0.0"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.params = config.mean_reversion

    def positions(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["close"]
        window = self.params.window
        mean = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std()
        z = (close - mean) / std.replace(0.0, np.nan)
        raw = pd.Series(0.0, index=frame.index)
        raw = raw.mask(z < -self.params.zscore, 1.0)
        raw = raw.mask(z > self.params.zscore, -1.0)
        raw = raw.where(z.notna(), 0.0)
        out = frame[["timestamp", "close"]].copy()
        out["position"] = apply_short_policy(raw, self.config.allow_short)
        out["zscore"] = z
        out["reason"] = [
            _reason(value, pos, self.params.zscore)
            for value, pos in zip(z.tolist(), out["position"].tolist(), strict=True)
        ]
        return out


def _reason(z: float, position: float, band: float) -> str:
    if pd.isna(z):
        return "Warmup: rolling z-score not available."
    if position > 0:
        return f"Close z-score {z:.3f} below -{band:.3f} (fade weakness)."
    if position < 0:
        return f"Close z-score {z:.3f} above {band:.3f} (fade strength)."
    return f"Close z-score {z:.3f} inside ±{band:.3f} (or shorts disabled)."
