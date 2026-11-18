from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_forex.config.schema import RiskConfig
from quant_forex.risk.manager import RiskManager
from quant_forex.risk.sizing import fixed_fraction_units, volatility_target_units


def test_max_risk_cannot_be_looser_than_target() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(risk_per_trade=0.01, max_risk_per_trade=0.005)


def test_exposure_cannot_exceed_leverage() -> None:
    with pytest.raises(ValidationError):
        RiskConfig(max_leverage=2.0, max_gross_exposure=5.0)


def test_high_vol_reduces_size_under_min_method() -> None:
    cfg = RiskConfig(sizing_method="min", risk_per_trade=0.005, max_risk_per_trade=0.005)
    mgr = RiskManager(cfg, lot_size=100_000)
    quiet = mgr.size(equity=10_000, entry=1.10, stop=1.09, realized_vol=0.002)
    noisy = mgr.size(equity=10_000, entry=1.10, stop=1.09, realized_vol=0.03)
    assert noisy.units < quiet.units
    assert noisy.units > 0


def test_volatility_method_scales_inversely_with_vol() -> None:
    low = volatility_target_units(10_000, 1.10, 0.005, 0.10, 252)
    high = volatility_target_units(10_000, 1.10, 0.02, 0.10, 252)
    assert high < low
    assert high == pytest.approx(low * (0.005 / 0.02))


def test_max_risk_cap_binds() -> None:
    cfg = RiskConfig(
        sizing_method="fixed_fraction",
        risk_per_trade=0.005,
        max_risk_per_trade=0.005,
        max_leverage=50.0,
        max_gross_exposure=50.0,
        max_position_size=100.0,
    )
    mgr = RiskManager(cfg, lot_size=100_000)
    decision = mgr.size(equity=10_000, entry=1.10, stop=1.09, realized_vol=None)
    assert decision.cash_at_risk == pytest.approx(10_000 * 0.005, rel=1e-6)
    assert "max_risk_per_trade" not in decision.capped_by or decision.units <= (
        10_000 * 0.005 / 0.01
    )


def test_concurrent_positions_block_entry() -> None:
    mgr = RiskManager(RiskConfig(max_concurrent_positions=1), lot_size=100_000)
    blocked = mgr.size(
        equity=10_000, entry=1.10, stop=1.09, realized_vol=0.01, open_positions=1
    )
    assert blocked.allowed is False
    assert blocked.units == 0.0
    assert blocked.reason == "max_concurrent_positions"


def test_missing_vol_does_not_inflate_min_size() -> None:
    cfg = RiskConfig(sizing_method="min")
    mgr = RiskManager(cfg, lot_size=100_000)
    with_vol = mgr.size(equity=10_000, entry=1.10, stop=1.09, realized_vol=0.004)
    no_vol = mgr.size(equity=10_000, entry=1.10, stop=1.09, realized_vol=None)
    fixed = fixed_fraction_units(10_000, 1.10, 1.09, cfg.risk_per_trade)
    assert no_vol.units <= fixed + 1e-9
    assert with_vol.units <= no_vol.units + 1e-9 or with_vol.units <= fixed


def test_drawdown_and_daily_loss_gates() -> None:
    mgr = RiskManager(RiskConfig(max_drawdown=0.10, max_daily_loss=0.02), lot_size=100_000)
    assert mgr.drawdown_breached(peak=10_000, equity=8_999) is True
    assert mgr.drawdown_breached(peak=10_000, equity=9_500) is False
    assert mgr.daily_loss_breached(10_000, 9_700) is True
    assert mgr.daily_loss_breached(10_000, 9_900) is False
