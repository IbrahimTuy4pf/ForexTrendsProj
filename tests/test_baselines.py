from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from quant_forex.config.schema import AppConfig, MACrossoverParams, StrategyConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.strategies.buy_and_hold import BuyAndHoldStrategy
from quant_forex.strategies.diagnostics import naive_next_bar_diagnostics
from quant_forex.strategies.ma_crossover import MovingAverageCrossoverStrategy
from quant_forex.strategies.momentum import MomentumStrategy
from quant_forex.strategies.pipeline import run_baselines
from quant_forex.strategies.signals_from_positions import build_entry_signals


def _uptrend(n: int = 40) -> pd.DataFrame:
    close = pd.Series([1.0 + 0.01 * i for i in range(n)])
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC"),
            "open": close,
            "high": close + 0.002,
            "low": close - 0.001,
            "close": close,
            "volume": 1.0,
        }
    )


def test_ma_fast_must_be_less_than_slow() -> None:
    with pytest.raises(ValidationError):
        MACrossoverParams(fast=20, slow=10)


def test_buy_and_hold_is_always_long() -> None:
    frame = generate_synthetic_ohlcv(n_rows=15, seed=21)
    pos = BuyAndHoldStrategy(StrategyConfig()).positions(frame)
    assert (pos["position"] == 1.0).all()


def test_ma_crossover_goes_long_in_uptrend() -> None:
    frame = _uptrend()
    cfg = StrategyConfig(ma_crossover={"fast": 3, "slow": 5})
    pos = MovingAverageCrossoverStrategy(cfg).positions(frame)
    assert (pos.loc[10:, "position"] == 1.0).all()


def test_short_policy_flattens_negative_positions() -> None:
    frame = generate_synthetic_ohlcv(n_rows=60, seed=22)
    with_short = MomentumStrategy(StrategyConfig(allow_short=True, momentum={"lookback": 5})).positions(frame)
    no_short = MomentumStrategy(StrategyConfig(allow_short=False, momentum={"lookback": 5})).positions(frame)
    assert (no_short["position"] >= 0).all()
    if (with_short["position"] < 0).any():
        assert (no_short["position"] == 0).sum() >= (with_short["position"] < 0).sum()


def test_future_bar_does_not_change_earlier_ma_positions() -> None:
    frame = generate_synthetic_ohlcv(n_rows=50, seed=23)
    cfg = StrategyConfig(ma_crossover={"fast": 5, "slow": 10})
    original = MovingAverageCrossoverStrategy(cfg).positions(frame)
    mutated = frame.copy()
    mutated.loc[mutated.index[-1], "close"] *= 1.2
    mutated.loc[mutated.index[-1], "high"] = max(
        mutated.loc[mutated.index[-1], "high"], mutated.loc[mutated.index[-1], "close"]
    )
    redone = MovingAverageCrossoverStrategy(cfg).positions(mutated)
    pd.testing.assert_series_equal(
        original["position"].iloc[:-1],
        redone["position"].iloc[:-1],
        check_names=False,
    )


def test_naive_diagnostics_use_next_bar_not_same_bar() -> None:
    frame = _uptrend(12)
    position = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    naive = naive_next_bar_diagnostics(frame, position, "test", n_entries=1)
    same_bar = float((position * frame["close"].pct_change()).sum())
    assert naive.naive_sum_next_return != pytest.approx(same_bar)


def test_entry_signals_have_live_prices_not_placeholders() -> None:
    frame = _uptrend(30)
    cfg = StrategyConfig(ma_crossover={"fast": 3, "slow": 5}, stop_atr_multiple=1.5, take_profit_atr_multiple=3.0)
    pos = MovingAverageCrossoverStrategy(cfg).positions(frame)
    signals = build_entry_signals(
        frame,
        pos,
        instrument="EURUSD",
        strategy_name="ma_crossover",
        strategy_version="1.0.0",
        atr_window=5,
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        risk_percentage=0.005,
    )
    assert signals
    first = signals[0]
    assert first.instrument == "EURUSD"
    assert first.direction == "LONG"
    assert first.entry == pytest.approx(float(frame.loc[pos.index[pos["position"] > 0][0], "close"]))
    assert first.stop_loss is not None and first.stop_loss < first.entry
    assert first.take_profit is not None and first.take_profit > first.entry
    assert first.risk_percentage == 0.005
    assert first.position_size is None
    assert "1.XXXX" not in first.reason


def test_baseline_pipeline_writes_outputs(project_paths) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=80, seed=24)
    frame.insert(0, "symbol", "EURUSD")
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    config = AppConfig()
    result = run_baselines(config, project_paths, input_csv=csv_path)
    assert result.signals_path.is_file()
    assert result.positions_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "not a backtest" in text.lower().replace("*", "")
    names = {run.name for run in result.runs}
    assert names == {"buy_and_hold", "ma_crossover", "momentum", "mean_reversion", "breakout"}
    positions = pd.read_csv(result.positions_path)
    assert set(positions["instrument"].unique()) == {"EURUSD"}
