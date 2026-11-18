"""Always-long FX pair stance. This is a benchmark, not advice.

Holding EURUSD long is equivalent to being long EUR vs USD, not an equity buy-and-hold.
"""

from __future__ import annotations

import pandas as pd

from quant_forex.config.schema import StrategyConfig


class BuyAndHoldStrategy:
    name = "buy_and_hold"
    version = "1.0.0"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def positions(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["timestamp", "close"]].copy()
        out["position"] = 1.0
        out["reason"] = "Always long the configured FX pair (benchmark, not a forecast)."
        return out
