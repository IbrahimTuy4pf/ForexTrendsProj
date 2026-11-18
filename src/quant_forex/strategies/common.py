"""Shared helpers for causal baseline strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_short_policy(position: pd.Series, allow_short: bool) -> pd.Series:
    if allow_short:
        return position.astype("float64")
    return position.clip(lower=0).astype("float64")


def true_range_atr(frame: pd.DataFrame, window: int) -> pd.Series:
    prev_close = frame["close"].shift(1)
    high_low = frame["high"] - frame["low"]
    high_prev = (frame["high"] - prev_close).abs()
    low_prev = (frame["low"] - prev_close).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def stops_from_atr(
    direction: pd.Series,
    entry: pd.Series,
    atr: pd.Series,
    stop_multiple: float | None,
    take_multiple: float | None,
) -> tuple[pd.Series, pd.Series]:
    stop = pd.Series(np.nan, index=direction.index, dtype="float64")
    take = pd.Series(np.nan, index=direction.index, dtype="float64")
    if stop_multiple is not None:
        long_stop = entry - stop_multiple * atr
        short_stop = entry + stop_multiple * atr
        stop = np.where(direction == 1, long_stop, np.where(direction == -1, short_stop, np.nan))
        stop = pd.Series(stop, index=direction.index, dtype="float64")
    if take_multiple is not None:
        long_tp = entry + take_multiple * atr
        short_tp = entry - take_multiple * atr
        take = np.where(direction == 1, long_tp, np.where(direction == -1, short_tp, np.nan))
        take = pd.Series(take, index=direction.index, dtype="float64")
    return stop, take
