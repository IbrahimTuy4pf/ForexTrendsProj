"""UTC-normalized timestamps for market data."""

from __future__ import annotations

import pandas as pd


def ensure_utc_datetime(series: pd.Series, timezone: str = "UTC") -> pd.Series:
    """Parse timestamps and convert to UTC.

    Naive timestamps are localized to ``timezone`` (not assumed UTC unless timezone is UTC).
    """
    parsed = pd.to_datetime(series, utc=False, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(timezone)
    return parsed.dt.tz_convert("UTC")
