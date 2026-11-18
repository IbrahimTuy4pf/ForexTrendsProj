from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_forex.backtesting.engine import run_backtest
from quant_forex.config.schema import AppConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.execution.broker import LiveTradingNotImplemented, submit_live_order
from quant_forex.execution.pipeline import run_paper
from quant_forex.strategies.buy_and_hold import BuyAndHoldStrategy


def test_live_order_cannot_be_submitted() -> None:
    with pytest.raises(LiveTradingNotImplemented):
        submit_live_order(symbol="EURUSD", units=1.0)


def test_paper_requires_opt_in(project_paths) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=40, seed=51)
    frame.to_csv(csv_path, index=False)
    config = AppConfig()
    config.execution.paper_trading = False
    with pytest.raises(PermissionError, match="opt-in"):
        run_paper(config, project_paths, input_csv=csv_path, strategy_name="buy_and_hold")


def test_flatten_at_end_false_keeps_open_units() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], utc=True
            ),
            "open": [1.10, 1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09, 1.09],
            "close": [1.10, 1.101, 1.102, 1.103],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    config = AppConfig()
    config.instrument.symbol = "EURUSD"
    config.costs.spread_pips = 1.0
    config.costs.slippage_pips = 0.2
    pos = pd.Series([1.0, 1.0, 1.0, 1.0])
    stop = pd.Series([1.08, 1.08, 1.08, 1.08])
    closed = run_backtest(frame, pos, stop=stop, take=None, config=config, strategy_name="u")
    opened = run_backtest(
        frame,
        pos,
        stop=stop,
        take=None,
        config=config,
        strategy_name="u",
        flatten_at_end=False,
    )
    assert closed.equity_curve["units"].iloc[-1] == 0.0
    assert opened.equity_curve["units"].iloc[-1] != 0.0
    assert any(trade.exit_reason == "end_of_data" for trade in closed.trades)
    assert not any(trade.exit_reason == "end_of_data" for trade in opened.trades)


def test_paper_session_writes_simulated_ledger(project_paths) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=80, seed=52)
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    config = AppConfig()
    config.execution.paper_trading = True
    config.paper.ledger_path = str(project_paths.data_processed / "paper_ledger.sqlite")
    config.strategies.momentum.enabled = False
    config.strategies.mean_reversion.enabled = False
    config.strategies.breakout.enabled = False
    config.strategies.ma_crossover.enabled = False
    result = run_paper(config, project_paths, input_csv=csv_path, strategy_name="buy_and_hold")
    assert result.ledger_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "SIMULATED" in text
    assert "No broker" in text or "no broker" in text.lower()
    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["broker"] is None
    assert data["live_trading"] is False
    assert data["venue"] == "paper_simulator"
    again = run_paper(config, project_paths, input_csv=csv_path, strategy_name="buy_and_hold")
    assert again.n_trades == result.n_trades
    held = BuyAndHoldStrategy(config.strategies).positions(frame)
    assert float(held["position"].iloc[-1]) == 1.0
    assert result.units != 0.0
