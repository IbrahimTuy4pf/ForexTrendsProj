"""Descriptive columns for charts and tables.

Close-to-close returns use only the previous close. They are not trading signals
and are not persisted as a feature store (that is Phase 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_forex.utils.sessions import (
    LONDON_HOURS,
    NEW_YORK_HOURS,
    OVERLAP_HOURS,
    sessions_applicable,
)


def add_descriptive_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if "close" not in frame.columns:
        raise ValueError("OHLCV frame must include close.")
    out = frame.copy()
    close = out["close"]
    out["simple_return"] = close.pct_change()
    out["log_return"] = np.log(close / close.shift(1))
    if {"high", "low"}.issubset(out.columns):
        out["high_low_range"] = out["high"] - out["low"]
        out["high_low_pct"] = out["high_low_range"] / close
    if {"open", "close", "high", "low"}.issubset(out.columns):
        out["candle_body"] = (out["close"] - out["open"]).abs()
        out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
        out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    stamps = pd.to_datetime(out["timestamp"], utc=True)
    out["hour_utc"] = stamps.dt.hour
    out["day_of_week"] = stamps.dt.day_name()
    out["weekday_number"] = stamps.dt.dayofweek
    out["session_applicable"] = sessions_applicable(stamps)
    out["london_session"] = out["hour_utc"].isin(LONDON_HOURS) & out["session_applicable"]
    out["new_york_session"] = out["hour_utc"].isin(NEW_YORK_HOURS) & out["session_applicable"]
    out["session_overlap"] = out["hour_utc"].isin(OVERLAP_HOURS) & out["session_applicable"]
    return out
