"""Position-size candidates. Smaller is treated as safer, not worse."""

from __future__ import annotations

import math


def stop_distance(entry: float, stop: float | None) -> float:
    if stop is None or abs(entry - stop) < 1e-12:
        return entry * 0.01
    return abs(entry - stop)


def fixed_fraction_units(equity: float, entry: float, stop: float | None, risk_fraction: float) -> float:
    if equity <= 0 or entry <= 0 or risk_fraction <= 0:
        return 0.0
    return (equity * risk_fraction) / stop_distance(entry, stop)


def volatility_target_units(
    equity: float,
    entry: float,
    realized_vol: float | None,
    annual_vol_target: float,
    periods_per_year: int,
) -> float:
    """Size so notional * daily vol ≈ equity * daily vol target.

    ``realized_vol`` is the causal close-to-close standard deviation (already daily).
    Missing or zero vol → no volatility size (do not invent a huge position).
    """
    if equity <= 0 or entry <= 0 or realized_vol is None or realized_vol <= 0:
        return 0.0
    daily_target = annual_vol_target / math.sqrt(periods_per_year)
    return (equity * daily_target) / (entry * realized_vol)
