"""Calendar gap detection relative to an expected bar frequency."""

from __future__ import annotations

import pandas as pd


def missing_bars(timestamps: pd.Series, expected_frequency: str | None = "B") -> pd.DatetimeIndex:
    """Return expected timestamps that are absent from ``timestamps``.

    For daily FX, ``B`` (business days) is the usual expectation. This does not
    invent prices for the missing days.
    """
    stamps = pd.to_datetime(timestamps, utc=True).dropna().sort_values()
    if stamps.empty or expected_frequency is None:
        return pd.DatetimeIndex([], tz="UTC")
    start, end = stamps.min(), stamps.max()
    expected = pd.date_range(start=start, end=end, freq=expected_frequency, tz="UTC")
    present = pd.DatetimeIndex(stamps).normalize()
    expected_norm = expected.normalize()
    return expected_norm.difference(present)


def gap_table(timestamps: pd.Series, expected_frequency: str | None = "B") -> pd.DataFrame:
    missing = missing_bars(timestamps, expected_frequency)
    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "n_missing_bars"])

    one_and_half_days_ns = 1.5 * 24 * 3600 * 1_000_000_000
    deltas = pd.Series(missing.asi8, index=missing).diff().fillna(0.0)
    group_id = (deltas > one_and_half_days_ns).cumsum()
    rows = []
    for _, part in pd.Series(1, index=missing).groupby(group_id):
        idx = part.index
        rows.append(
            {
                "gap_start": idx.min().isoformat(),
                "gap_end": idx.max().isoformat(),
                "n_missing_bars": int(len(idx)),
            }
        )
    return pd.DataFrame(rows)
