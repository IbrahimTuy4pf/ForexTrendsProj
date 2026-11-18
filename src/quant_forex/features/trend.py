"""Causal moving-average and MACD features."""

from __future__ import annotations

import pandas as pd

from quant_forex.config.schema import FeatureConfig


def add_trend_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    sma_cols: list[str] = []
    for window in config.sma_windows:
        name = f"sma_{window}"
        out[name] = close.rolling(window=window, min_periods=window).mean()
        out[f"close_sma_{window}_ratio"] = close / out[name]
        out[f"close_above_sma_{window}"] = (close > out[name]).astype("float64")
        sma_cols.append(name)
    for window in config.ema_windows:
        name = f"ema_{window}"
        out[name] = close.ewm(span=window, adjust=False, min_periods=window).mean()
        out[f"close_ema_{window}_ratio"] = close / out[name]
    if len(sma_cols) >= 2:
        fast, slow = sma_cols[0], sma_cols[1]
        out[f"{fast}_minus_{slow}"] = out[fast] - out[slow]
        out[f"{fast}_above_{slow}"] = (out[fast] > out[slow]).astype("float64")
    fast = close.ewm(span=config.macd_fast, adjust=False, min_periods=config.macd_fast).mean()
    slow = close.ewm(span=config.macd_slow, adjust=False, min_periods=config.macd_slow).mean()
    out["macd"] = fast - slow
    out["macd_signal"] = out["macd"].ewm(
        span=config.macd_signal, adjust=False, min_periods=config.macd_signal
    ).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out
