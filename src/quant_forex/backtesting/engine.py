"""Bar-based FX backtester. Signals at close t fill at open t+1. Costs are never assumed zero."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.backtesting.fills import (
    entry_fill,
    exit_fill,
    half_spread_price,
    slippage_price,
    stop_fill,
    take_fill,
)
from quant_forex.backtesting.sizing import commission_cost, pnl_account_currency
from quant_forex.risk.manager import RiskManager


@dataclass
class Trade:
    strategy: str
    direction: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    costs: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    strategy: str
    initial_capital: float
    equity_curve: pd.DataFrame
    trades: list[Trade]
    halted: bool
    halt_reason: str | None
    assumptions: dict[str, object] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        if self.equity_curve.empty:
            return self.initial_capital
        return float(self.equity_curve["equity"].iloc[-1])


def run_backtest(
    ohlcv: pd.DataFrame,
    position: pd.Series,
    *,
    stop: pd.Series | None,
    take: pd.Series | None,
    config: AppConfig,
    strategy_name: str,
    realized_vol: pd.Series | None = None,
    flatten_at_end: bool = True,
) -> BacktestResult:
    if config.backtest.fill_on != "next_open":
        raise ValueError("Only fill_on=next_open is implemented (avoids same-bar close fills).")

    frame = ohlcv.sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    pos_series = pd.Series(position).reset_index(drop=True)
    if len(pos_series) != n:
        raise ValueError("position series length must match OHLCV rows.")
    pos = pos_series.fillna(0.0).to_numpy(dtype=float)
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    dates = stamps.dt.date.to_numpy()
    ts_text = [s.isoformat() if hasattr(s, "isoformat") else str(s) for s in stamps]

    stop_arr = (
        pd.Series(stop).reset_index(drop=True).to_numpy(dtype=float)
        if stop is not None
        else np.full(n, np.nan)
    )
    take_arr = (
        pd.Series(take).reset_index(drop=True).to_numpy(dtype=float)
        if take is not None
        else np.full(n, np.nan)
    )
    if len(stop_arr) != n or len(take_arr) != n:
        raise ValueError("stop/take series length must match OHLCV rows.")

    vol_arr = (
        pd.Series(realized_vol).reset_index(drop=True).to_numpy(dtype=float)
        if realized_vol is not None
        else np.full(n, np.nan)
    )
    if len(vol_arr) != n:
        raise ValueError("realized_vol length must match OHLCV rows.")

    half_spread = half_spread_price(config.costs.spread_pips, config.instrument.pip_size)
    slip = slippage_price(config.costs.slippage_pips, config.instrument.pip_size)
    if config.costs.spread_pips < 0 or config.costs.slippage_pips < 0:
        raise ValueError("Spread and slippage cannot be negative.")

    base = config.instrument.base_currency
    quote = config.instrument.quote_currency
    account = config.backtest.account_currency
    initial = config.backtest.initial_capital
    realized = 0.0
    peak = initial
    day_start = initial
    current_day: date | None = None
    blocked_today = False
    risk_mgr = RiskManager(config.risk, config.backtest.lot_size)
    halted = False
    halt_reason: str | None = None

    units = 0.0
    direction = 0
    entry_price = 0.0
    entry_idx = -1
    live_stop = np.nan
    live_take = np.nan
    entry_costs = 0.0

    trades: list[Trade] = []
    equity_rows: list[dict[str, object]] = []

    def mark_equity(t: int) -> float:
        unreal = 0.0
        if units != 0:
            unreal = pnl_account_currency(
                units, entry_price, closes[t], base=base, quote=quote, account=account
            )
        return initial + realized + unreal

    def close_position(t: int, price: float, reason: str) -> None:
        nonlocal units, direction, realized, entry_costs, live_stop, live_take
        if units == 0:
            return
        exit_comm = commission_cost(units, config.backtest.lot_size, config.costs.commission_per_lot)
        pnl = pnl_account_currency(
            units, entry_price, price, base=base, quote=quote, account=account
        )
        costs = entry_costs + exit_comm
        net = pnl - exit_comm
        realized += net
        trades.append(
            Trade(
                strategy=strategy_name,
                direction=direction,
                entry_time=ts_text[entry_idx],
                exit_time=ts_text[t],
                entry_price=entry_price,
                exit_price=price,
                units=units,
                pnl=net,
                costs=costs,
                exit_reason=reason,
                bars_held=t - entry_idx,
            )
        )
        units = 0.0
        direction = 0
        live_stop = np.nan
        live_take = np.nan
        entry_costs = 0.0

    def open_position(t: int, desired: int, signal_idx: int) -> None:
        nonlocal units, direction, entry_price, entry_idx, live_stop, live_take, entry_costs, realized
        equity_now = initial + realized if not equity_rows else float(equity_rows[-1]["equity"])
        sig_stop = stop_arr[signal_idx]
        fill = entry_fill(opens[t], desired, half_spread, slip)
        stop_for_size = None if np.isnan(sig_stop) else float(sig_stop)
        vol_now = float(vol_arr[signal_idx]) if signal_idx >= 0 else float("nan")
        if vol_now != vol_now:
            vol_now = None
        decision = risk_mgr.size(
            equity=equity_now,
            entry=fill,
            stop=stop_for_size,
            realized_vol=vol_now,
            current_notional=0.0,
            open_positions=0 if abs(units) < 1e-12 else 1,
        )
        qty = decision.units if decision.allowed else 0.0
        if qty <= 0:
            return
        signed = qty if desired > 0 else -qty
        comm = commission_cost(signed, config.backtest.lot_size, config.costs.commission_per_lot)
        realized -= comm
        units = signed
        direction = desired
        entry_price = fill
        entry_idx = t
        entry_costs = comm
        live_stop = sig_stop
        live_take = take_arr[signal_idx]

    for t in range(n):
        if current_day != dates[t]:
            current_day = dates[t]
            day_start = initial + realized if not equity_rows else float(equity_rows[-1]["equity"])
            blocked_today = False

        signal_idx = t - 1
        desired = 0 if t == 0 else int(np.sign(pos[signal_idx]))

        if halted and units != 0:
            px = exit_fill(opens[t], direction, half_spread, slip)
            close_position(t, px, halt_reason or "halt")

        cur = 0 if units == 0 else (1 if units > 0 else -1)
        if not halted and cur != 0 and desired != cur:
            px = exit_fill(opens[t], direction, half_spread, slip)
            close_position(t, px, "signal")
            cur = 0

        if (
            not halted
            and not blocked_today
            and cur == 0
            and desired != 0
            and abs(units) < 1e-12
        ):
            ok, _reason = risk_mgr.can_open(0)
            if ok:
                open_position(t, desired, signal_idx)

        if units != 0 and not np.isnan(live_stop):
            long = units > 0
            hit_stop = lows[t] <= live_stop if long else highs[t] >= live_stop
            hit_take = False
            if not np.isnan(live_take):
                hit_take = highs[t] >= live_take if long else lows[t] <= live_take
            if hit_stop and hit_take:
                px = stop_fill(
                    direction=direction,
                    open_price=opens[t],
                    stop=float(live_stop),
                    half_spread=half_spread,
                    slip=slip,
                )
                close_position(t, px, "stop_same_bar_priority")
            elif hit_stop:
                px = stop_fill(
                    direction=direction,
                    open_price=opens[t],
                    stop=float(live_stop),
                    half_spread=half_spread,
                    slip=slip,
                )
                close_position(t, px, "stop")
            elif hit_take:
                px = take_fill(
                    direction=direction,
                    open_price=opens[t],
                    take=float(live_take),
                    half_spread=half_spread,
                    slip=slip,
                )
                close_position(t, px, "take_profit")

        equity = mark_equity(t)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        day_loss = (day_start - equity) / day_start if day_start > 0 else 0.0
        if not halted and risk_mgr.drawdown_breached(peak, equity):
            halted = True
            halt_reason = "max_drawdown"
        if not blocked_today and risk_mgr.daily_loss_breached(day_start, equity):
            blocked_today = True
            if units != 0:
                px = exit_fill(closes[t], direction, half_spread, slip)
                close_position(t, px, "max_daily_loss")
                equity = mark_equity(t)
        equity_rows.append(
            {
                "timestamp": stamps.iloc[t],
                "equity": equity,
                "units": units,
                "drawdown": dd,
                "halted": halted,
            }
        )

    if flatten_at_end and units != 0:
        px = exit_fill(closes[-1], direction, half_spread, slip)
        close_position(n - 1, px, "end_of_data")
        equity_rows[-1]["equity"] = mark_equity(n - 1)
        equity_rows[-1]["units"] = 0.0

    return BacktestResult(
        strategy=strategy_name,
        initial_capital=initial,
        equity_curve=pd.DataFrame(equity_rows),
        trades=trades,
        halted=halted,
        halt_reason=halt_reason,
        assumptions={
            "fill_on": config.backtest.fill_on,
            "spread_pips": config.costs.spread_pips,
            "slippage_pips": config.costs.slippage_pips,
            "commission_per_lot": config.costs.commission_per_lot,
            "same_bar_stop_take": config.backtest.same_bar_stop_take,
            "risk_per_trade": config.risk.risk_per_trade,
            "max_leverage": config.risk.max_leverage,
            "max_daily_loss": config.risk.max_daily_loss,
            "max_drawdown": config.risk.max_drawdown,
            "sizing_method": config.risk.sizing_method,
            "annual_vol_target": config.risk.annual_vol_target,
            "max_gross_exposure": config.risk.max_gross_exposure,
            "max_risk_per_trade": config.risk.max_risk_per_trade,
            "note": "Costs applied. Sizing uses the tightest risk cap, not maximum return.",
            "flatten_at_end": flatten_at_end,
            "venue": "research_simulator",
        },
    )
