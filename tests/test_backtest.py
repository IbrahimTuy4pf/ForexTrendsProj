from __future__ import annotations

import pandas as pd
import pytest

from quant_forex.backtesting.engine import run_backtest
from quant_forex.backtesting.fills import entry_fill, half_spread_price, slippage_price
from quant_forex.backtesting.pipeline import run_backtest_pipeline
from quant_forex.backtesting.sizing import size_base_units
from quant_forex.config.schema import AppConfig
from quant_forex.data.sample import generate_synthetic_ohlcv


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], utc=True
            ),
            "open": [1.1000, 1.1000, 1.1000, 1.1000],
            "high": [1.1010, 1.1010, 1.2000, 1.1010],
            "low": [1.0990, 1.0990, 1.0500, 1.0990],
            "close": [1.1000, 1.1005, 1.0900, 1.1000],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _config(**risk: float) -> AppConfig:
    cfg = AppConfig()
    cfg.instrument.symbol = "EURUSD"
    cfg.instrument.base_currency = "EUR"
    cfg.instrument.quote_currency = "USD"
    cfg.instrument.pip_size = 0.0001
    cfg.costs.spread_pips = 1.0
    cfg.costs.slippage_pips = 0.2
    cfg.costs.commission_per_lot = 0.0
    cfg.backtest.initial_capital = 10_000.0
    for key, value in risk.items():
        setattr(cfg.risk, key, value)
    return cfg


def test_long_entry_is_worse_than_mid_open() -> None:
    cfg = _config()
    half = half_spread_price(cfg.costs.spread_pips, cfg.instrument.pip_size)
    slip = slippage_price(cfg.costs.slippage_pips, cfg.instrument.pip_size)
    fill = entry_fill(1.1000, 1, half, slip)
    assert fill > 1.1000


def test_signal_fills_next_open_not_signal_bar() -> None:
    frame = _bars()
    position = pd.Series([1.0, 1.0, 1.0, 0.0])
    stop = pd.Series([1.0800, 1.0800, 1.0800, 1.0800])
    result = run_backtest(
        frame, position, stop=stop, take=None, config=_config(), strategy_name="unit"
    )
    assert result.trades
    assert result.trades[0].entry_time.startswith("2024-01-03")


def test_stop_exits_when_low_breaks_level() -> None:
    frame = _bars()
    position = pd.Series([1.0, 1.0, 1.0, 1.0])
    stop = pd.Series([1.0800, 1.0800, 1.0800, 1.0800])
    result = run_backtest(
        frame, position, stop=stop, take=None, config=_config(), strategy_name="unit"
    )
    reasons = [trade.exit_reason for trade in result.trades]
    assert "stop" in reasons or "stop_same_bar_priority" in reasons


def test_same_bar_stop_and_target_uses_stop() -> None:
    frame = _bars()
    position = pd.Series([1.0, 1.0, 1.0, 1.0])
    stop = pd.Series([1.0800, 1.0800, 1.0800, 1.0800])
    take = pd.Series([1.1200, 1.1200, 1.1200, 1.1200])
    result = run_backtest(
        frame, position, stop=stop, take=take, config=_config(), strategy_name="unit"
    )
    assert any(trade.exit_reason == "stop_same_bar_priority" for trade in result.trades)


def test_costs_reduce_equity_versus_zero_spread() -> None:
    frame = _bars()
    position = pd.Series([1.0, 1.0, 0.0, 0.0])
    costly = _config()
    free = _config()
    free.costs.spread_pips = 0.0
    free.costs.slippage_pips = 0.0
    a = run_backtest(frame, position, stop=None, take=None, config=costly, strategy_name="c")
    b = run_backtest(frame, position, stop=None, take=None, config=free, strategy_name="f")
    assert a.final_equity <= b.final_equity


def test_leverage_and_lot_caps_bind() -> None:
    units = size_base_units(
        equity=10_000,
        entry=1.10,
        stop=1.09,
        risk_fraction=0.50,
        max_leverage=2.0,
        max_lots=0.1,
        lot_size=100_000,
    )
    assert units == pytest.approx(0.1 * 100_000)


def test_max_drawdown_halts_new_risk() -> None:
    close = [1.10, 1.09, 1.08, 1.07, 1.06, 1.05]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC"),
            "open": close,
            "high": [c + 0.0005 for c in close],
            "low": [c - 0.0005 for c in close],
            "close": close,
            "volume": 1.0,
        }
    )
    cfg = _config(max_drawdown=0.001)
    result = run_backtest(
        frame,
        pd.Series([1.0] * 6),
        stop=None,
        take=None,
        config=cfg,
        strategy_name="dd",
    )
    assert result.halted
    assert result.halt_reason == "max_drawdown"


def test_backtest_pipeline_writes_files(project_paths) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=90, seed=32)
    frame.insert(0, "symbol", "EURUSD")
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    config = AppConfig()
    config.strategies.buy_and_hold.enabled = True
    config.strategies.ma_crossover.enabled = False
    config.strategies.momentum.enabled = False
    config.strategies.mean_reversion.enabled = False
    config.strategies.breakout.enabled = False
    result = run_backtest_pipeline(config, project_paths, input_csv=csv_path)
    assert result.report_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "next bar open" in text.lower()
    assert result.trades_path.is_file()
    assert (result.figure_dir / "buy_and_hold_equity.png").is_file()
