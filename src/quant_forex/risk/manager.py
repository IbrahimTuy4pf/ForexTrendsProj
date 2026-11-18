"""Risk manager: size orders by the *tightest* applicable cap, then gate new risk."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_forex.config.schema import RiskConfig
from quant_forex.risk.sizing import fixed_fraction_units, stop_distance, volatility_target_units


@dataclass
class SizingDecision:
    units: float
    method: str
    capped_by: list[str] = field(default_factory=list)
    notional: float = 0.0
    leverage: float = 0.0
    cash_at_risk: float = 0.0
    allowed: bool = True
    reason: str = ""


class RiskManager:
    def __init__(self, config: RiskConfig, lot_size: float) -> None:
        self.config = config
        self.lot_size = lot_size

    def drawdown_breached(self, peak: float, equity: float) -> bool:
        if peak <= 0:
            return False
        return (peak - equity) / peak >= self.config.max_drawdown

    def daily_loss_breached(self, day_start_equity: float, equity: float) -> bool:
        if day_start_equity <= 0:
            return False
        return (day_start_equity - equity) / day_start_equity >= self.config.max_daily_loss

    def can_open(self, open_positions: int) -> tuple[bool, str]:
        if open_positions >= self.config.max_concurrent_positions:
            return False, "max_concurrent_positions"
        return True, "ok"

    def size(
        self,
        *,
        equity: float,
        entry: float,
        stop: float | None,
        realized_vol: float | None,
        current_notional: float = 0.0,
        open_positions: int = 0,
    ) -> SizingDecision:
        allowed, reason = self.can_open(open_positions)
        if not allowed:
            return SizingDecision(units=0.0, method=self.config.sizing_method, allowed=False, reason=reason)
        if equity <= 0 or entry <= 0:
            return SizingDecision(units=0.0, method=self.config.sizing_method, allowed=False, reason="non_positive_equity")

        fixed = fixed_fraction_units(equity, entry, stop, self.config.risk_per_trade)
        vol = volatility_target_units(
            equity,
            entry,
            realized_vol,
            self.config.annual_vol_target,
            self.config.periods_per_year,
        )
        method = self.config.sizing_method
        if method == "fixed_fraction":
            raw = fixed
        elif method == "volatility":
            raw = vol
        else:
            candidates = [value for value in (fixed, vol) if value > 0]
            raw = min(candidates) if candidates else 0.0
        if raw <= 0:
            return SizingDecision(units=0.0, method=method, allowed=False, reason="no_valid_size")

        units, caps = self._apply_caps(raw, equity, entry, stop, current_notional)
        notional = units * entry
        leverage = notional / equity if equity else 0.0
        cash_at_risk = units * stop_distance(entry, stop)
        return SizingDecision(
            units=units,
            method=method,
            capped_by=caps,
            notional=notional,
            leverage=leverage,
            cash_at_risk=cash_at_risk,
            allowed=units > 0,
            reason="ok" if units > 0 else "capped_to_zero",
        )

    def _apply_caps(
        self,
        units: float,
        equity: float,
        entry: float,
        stop: float | None,
        current_notional: float,
    ) -> tuple[float, list[str]]:
        caps: list[str] = []
        dist = stop_distance(entry, stop)
        max_risk_units = (equity * self.config.max_risk_per_trade) / dist if dist else 0.0
        max_lev_units = (equity * self.config.max_leverage) / entry
        max_exp_units = max(0.0, (equity * self.config.max_gross_exposure - current_notional) / entry)
        max_lot_units = self.config.max_position_size * self.lot_size
        limited = units
        if limited > max_risk_units + 1e-12:
            limited = max_risk_units
            caps.append("max_risk_per_trade")
        if limited > max_lev_units + 1e-12:
            limited = max_lev_units
            caps.append("max_leverage")
        if limited > max_exp_units + 1e-12:
            limited = max_exp_units
            caps.append("max_gross_exposure")
        if limited > max_lot_units + 1e-12:
            limited = max_lot_units
            caps.append("max_position_size")
        return float(max(0.0, limited)), caps
