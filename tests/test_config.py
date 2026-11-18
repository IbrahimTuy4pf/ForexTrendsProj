from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant_forex.config.loader import load_config
from quant_forex.config.schema import AppConfig, ExecutionEnvSettings
from quant_forex.data.instruments import MAJOR_PAIRS


def test_default_symbol_is_configurable_not_baked_into_logic() -> None:
    gbp = AppConfig.model_validate(
        {"instrument": {"symbol": "GBP/USD", "yfinance_ticker": None}}
    )
    assert gbp.instrument.symbol == "GBPUSD"
    assert gbp.instrument.resolved_yfinance_ticker() == "GBPUSD=X"


def test_live_trading_rejected_in_yaml() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"execution": {"live_trading": True}})


def test_live_trading_rejected_in_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(ValidationError):
        ExecutionEnvSettings()


def test_load_config_from_yaml(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("instrument:\n  symbol: USDJPY\n", encoding="utf-8")
    config = load_config(path)
    assert config.instrument.symbol == "USDJPY"
    assert config.instrument.pip_size == 0.0001


def test_majors_catalog_includes_future_pairs() -> None:
    assert "EURUSD" in MAJOR_PAIRS
    assert "USDJPY" in MAJOR_PAIRS
    assert len(MAJOR_PAIRS) == 7
