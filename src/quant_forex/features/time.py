"""Calendar and session features from the bar timestamp only (no price look-ahead)."""

from __future__ import annotations

import pandas as pd

from quant_forex.utils.sessions import (
    LONDON_HOURS,
    NEW_YORK_HOURS,
    OVERLAP_HOURS,
    sessions_applicable,
)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    stamps = pd.to_datetime(out["timestamp"], utc=True)
    out["hour_utc"] = stamps.dt.hour.astype("float64")
    out["weekday_number"] = stamps.dt.dayofweek.astype("float64")
    out["is_weekend"] = (stamps.dt.dayofweek >= 5).astype("float64")
    applicable = sessions_applicable(stamps)
    out["session_applicable"] = applicable.astype("float64")
    out["london_session"] = (stamps.dt.hour.isin(list(LONDON_HOURS)) & applicable).astype("float64")
    out["new_york_session"] = (
        stamps.dt.hour.isin(list(NEW_YORK_HOURS)) & applicable
    ).astype("float64")
    out["session_overlap"] = (
        stamps.dt.hour.isin(list(OVERLAP_HOURS)) & applicable
    ).astype("float64")
    return out
