"""Causal price features. All rolling windows include the current bar and no future bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_forex.config.schema import FeatureConfig


def add_price_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    out["ret_1"] = close.pct_change()
    out["log_ret_1"] = np.log(close / close.shift(1))
    for window in config.rolling_return_windows:
        out[f"ret_{window}"] = close.pct_change(periods=window)
        out[f"log_ret_{window}"] = np.log(close / close.shift(window))
    if {"high", "low"}.issubset(out.columns):
        out["high_low_range"] = out["high"] - out["low"]
        out["high_low_pct"] = out["high_low_range"] / close.replace(0, np.nan)
    if {"open", "close", "high", "low"}.issubset(out.columns):
        out["candle_body"] = (out["close"] - out["open"]).abs()
        out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
        out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
        out["candle_direction"] = np.sign(out["close"] - out["open"])
    return out
