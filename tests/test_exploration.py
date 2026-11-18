from __future__ import annotations

import pandas as pd
import pytest

from quant_forex.config.schema import AppConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.exploration.gaps import missing_bars
from quant_forex.exploration.pipeline import run_exploration
from quant_forex.exploration.prepare import add_descriptive_columns
from quant_forex.exploration.stats import rolling_return_volatility


def test_returns_use_only_previous_close() -> None:
    frame = generate_synthetic_ohlcv(n_rows=5, seed=3)
    out = add_descriptive_columns(frame)
    assert pd.isna(out.loc[0, "simple_return"])
    expected = frame.loc[2, "close"] / frame.loc[1, "close"] - 1.0
    assert out.loc[2, "simple_return"] == pytest.approx(expected)


def test_rolling_vol_does_not_use_future_returns() -> None:
    returns = pd.Series([0.01] * 20)
    returns.iloc[19] = 5.0
    window = 5
    vol = rolling_return_volatility(returns, window)
    # Index 14 uses bars 10–14 only. The spike at 19 must not leak backward.
    assert vol.iloc[14] == pytest.approx(returns.iloc[10:15].std(ddof=1))
    assert vol.iloc[14] == pytest.approx(0.0, abs=1e-12)


def test_daily_bars_do_not_claim_sessions() -> None:
    frame = add_descriptive_columns(generate_synthetic_ohlcv(n_rows=10, seed=5))
    assert not frame["session_applicable"].any()
    assert not frame["london_session"].any()


def test_hourly_bars_flag_london_and_new_york() -> None:
    stamps = pd.date_range("2024-01-02 07:00", periods=13, freq="h", tz="UTC")
    close = 1.08 + pd.Series(range(13), dtype=float) * 0.0001
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "open": close,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": 1.0,
        }
    )
    out = add_descriptive_columns(frame)
    assert out["session_applicable"].all()
    assert bool(out.loc[out["hour_utc"] == 7, "london_session"].iloc[0])
    assert bool(out.loc[out["hour_utc"] == 14, "session_overlap"].iloc[0])
    assert bool(out.loc[out["hour_utc"] == 18, "new_york_session"].iloc[0])
    assert not bool(out.loc[out["hour_utc"] == 18, "london_session"].iloc[0])


def test_gap_detection_finds_removed_business_day() -> None:
    frame = generate_synthetic_ohlcv(n_rows=15, start="2024-01-02", freq="B", seed=6)
    removed = frame.iloc[7]["timestamp"]
    hole = frame.drop(index=7)
    missing = missing_bars(hole["timestamp"], "B")
    assert removed.normalize() in missing.normalize()


def test_exploration_writes_report_and_figures(project_paths, tmp_path) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_test_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=40, seed=8)
    frame.insert(0, "symbol", "EURUSD")
    frame.insert(1, "timeframe", "1d")
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    csv_path.write_text(
        "# SYNTHETIC DATA — not real EURUSD.\n" + export.to_csv(index=False),
        encoding="utf-8",
    )
    config = AppConfig()
    result = run_exploration(config, project_paths, input_csv=csv_path)
    assert result.n_rows == 40
    assert result.report_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "not a backtest" in text.lower() or "not a trading strategy" in text.lower()
    assert (result.figure_dir / "close_price.png").is_file()
    assert (result.figure_dir / "simple_returns_hist.png").is_file()
    assert result.summary.symbol == "EURUSD"
    assert result.summary.sessions_applicable is False
