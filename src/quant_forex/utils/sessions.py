"""Approximate FX session windows in UTC.

These are research approximations, not exchange calendars. Daily bars timestamped
at 00:00 UTC cannot identify a session.
"""

from __future__ import annotations

import pandas as pd

LONDON_HOURS = range(7, 16)
NEW_YORK_HOURS = range(12, 21)
OVERLAP_HOURS = range(12, 16)


def sessions_applicable(stamps: pd.Series) -> pd.Series:
    hours = pd.to_datetime(stamps, utc=True).dt.hour
    unique_hours = set(hours.dropna().unique().tolist())
    if unique_hours <= {0}:
        return pd.Series(False, index=stamps.index)
    return pd.Series(True, index=stamps.index)
