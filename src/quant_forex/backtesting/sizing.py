"""Position sizing with risk fraction, leverage, and lot caps."""

from __future__ import annotations


def size_base_units(
    *,
    equity: float,
    entry: float,
    stop: float | None,
    risk_fraction: float,
    max_leverage: float,
    max_lots: float,
    lot_size: float,
) -> float:
    if equity <= 0 or entry <= 0:
        return 0.0
    if stop is None or abs(entry - stop) < 1e-12:
        stop_distance = entry * 0.01
    else:
        stop_distance = abs(entry - stop)
    risk_cash = equity * risk_fraction
    units_from_risk = risk_cash / stop_distance
    units_from_leverage = (equity * max_leverage) / entry
    units_from_lots = max_lots * lot_size
    return float(max(0.0, min(units_from_risk, units_from_leverage, units_from_lots)))


def pnl_account_currency(
    units: float,
    entry: float,
    exit_price: float,
    *,
    base: str,
    quote: str,
    account: str,
) -> float:
    """PnL for a position sized in base-currency units.

    Quote-denominated accounts (EURUSD in USD) use units * delta.
    USDJPY in a USD account converts JPY PnL at the exit rate.
    """
    raw_quote = units * (exit_price - entry)
    account = account.upper()
    base = base.upper()
    quote = quote.upper()
    if quote == account:
        return raw_quote
    if base == account:
        return raw_quote / exit_price
    raise ValueError(
        f"Cannot convert PnL from {base}/{quote} into {account} without a cross-rate feed."
    )


def commission_cost(units: float, lot_size: float, commission_per_lot: float) -> float:
    if lot_size <= 0:
        return 0.0
    return abs(units) / lot_size * commission_per_lot
