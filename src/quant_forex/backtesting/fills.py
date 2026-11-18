"""Conservative FX fill prices: spread and slippage always widen the round trip."""

from __future__ import annotations


def half_spread_price(spread_pips: float, pip_size: float) -> float:
    return (spread_pips * pip_size) / 2.0


def slippage_price(slippage_pips: float, pip_size: float) -> float:
    return slippage_pips * pip_size


def entry_fill(mid: float, direction: int, half_spread: float, slip: float) -> float:
    """direction +1 long, -1 short. Long buys the ask; short sells the bid."""
    if direction > 0:
        return mid + half_spread + slip
    return mid - half_spread - slip


def exit_fill(mid: float, direction: int, half_spread: float, slip: float) -> float:
    """Close a long at the bid; close a short at the ask."""
    if direction > 0:
        return mid - half_spread - slip
    return mid + half_spread + slip


def stop_fill(
    *,
    direction: int,
    open_price: float,
    stop: float,
    half_spread: float,
    slip: float,
) -> float:
    """If the bar gaps through the stop, fill at the open (still with costs)."""
    if direction > 0:
        if open_price <= stop:
            return exit_fill(open_price, direction, half_spread, slip)
        return stop - slip
    if open_price >= stop:
        return exit_fill(open_price, direction, half_spread, slip)
    return stop + slip


def take_fill(
    *,
    direction: int,
    open_price: float,
    take: float,
    half_spread: float,
    slip: float,
) -> float:
    if direction > 0:
        if open_price >= take:
            return exit_fill(open_price, direction, half_spread, slip)
        return take - slip
    if open_price <= take:
        return exit_fill(open_price, direction, half_spread, slip)
    return take + slip
