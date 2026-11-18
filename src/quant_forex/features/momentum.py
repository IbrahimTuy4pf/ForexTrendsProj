"""Causal momentum features (RSI, ROC, momentum)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_forex.config.schema import FeatureConfig


def add_momentum_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    out[f"roc_{config.roc_window}"] = close.pct_change(periods=config.roc_window)
    out[f"momentum_{config.momentum_window}"] = close - close.shift(config.momentum_window)
    out[f"rsi_{config.rsi_window}"] = _wilder_rsi(close, config.rsi_window)
    return out


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))
