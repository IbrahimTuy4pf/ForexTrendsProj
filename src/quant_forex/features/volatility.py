"""Causal volatility features. Regime labels use rolling history, never the full sample."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_forex.config.schema import FeatureConfig


def add_volatility_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"]
    returns = close.pct_change()
    out[f"vol_{config.vol_window}"] = returns.rolling(
        window=config.vol_window, min_periods=config.vol_window
    ).std()
    if {"high", "low", "close"}.issubset(out.columns):
        out[f"atr_{config.atr_window}"] = _atr(out, config.atr_window)
    mid = close.rolling(window=config.bb_window, min_periods=config.bb_window).mean()
    std = close.rolling(window=config.bb_window, min_periods=config.bb_window).std()
    upper = mid + config.bb_std * std
    lower = mid - config.bb_std * std
    out[f"bb_mid_{config.bb_window}"] = mid
    out[f"bb_width_{config.bb_window}"] = (upper - lower) / mid.replace(0, np.nan)
    vol_col = f"vol_{config.vol_window}"
    vol_median = out[vol_col].rolling(
        window=config.vol_regime_window, min_periods=config.vol_regime_window
    ).median()
    out["vol_regime_high"] = (out[vol_col] > vol_median).astype("float64")
    out.loc[vol_median.isna(), "vol_regime_high"] = np.nan
    return out


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    high_low = frame["high"] - frame["low"]
    high_prev = (frame["high"] - prev_close).abs()
    low_prev = (frame["low"] - prev_close).abs()
    return pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)


def _atr(frame: pd.DataFrame, window: int) -> pd.Series:
    return _true_range(frame).rolling(window=window, min_periods=window).mean()
