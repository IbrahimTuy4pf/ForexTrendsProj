from __future__ import annotations

import pandas as pd

from quant_forex.data.quality import build_quality_report
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.data.schema import standardize_ohlcv_columns
from quant_forex.data.transforms import TransformationLog
from quant_forex.data.validation import DataCleaner
from quant_forex.config.schema import DataConfig


def test_column_aliases() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02"],
            "Open": [1.1],
            "High": [1.2],
            "Low": [1.0],
            "Close": [1.15],
            "Volume": [10],
        }
    )
    frame, mapping = standardize_ohlcv_columns(raw)
    assert "timestamp" in frame.columns
    assert mapping["Date"] == "timestamp"
    assert mapping["Open"] == "open"


def test_quality_report_detects_duplicates_and_bad_ohlc() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03"], utc=True
            ),
            "open": [1.10, 1.10, 1.11],
            "high": [1.09, 1.09, 1.12],  # high < open on first two rows
            "low": [1.08, 1.08, 1.10],
            "close": [1.085, 1.085, 1.115],
            "volume": [1, 1, 1],
        }
    )
    report = build_quality_report(
        frame,
        symbol="EURUSD",
        timeframe="1d",
        source="test",
        timezone="UTC",
        expected_frequency="1D",
        abnormal_return_threshold=0.05,
    )
    codes = {issue.code for issue in report.issues}
    assert "duplicate_timestamps" in codes
    assert "inconsistent_ohlc" in codes
    assert report.has_errors


def test_cleaner_sorts_and_drops_duplicates_with_log() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-03", "2024-01-01", "2024-01-01"],
            "open": [1.2, 1.1, 1.1],
            "high": [1.21, 1.12, 1.12],
            "low": [1.19, 1.09, 1.09],
            "close": [1.20, 1.11, 1.11],
            "volume": [1, 1, 1],
        }
    )
    log = TransformationLog()
    cleaner = DataCleaner(DataConfig(drop_duplicates=True, sort_chronologically=True), log)
    cleaned = cleaner.clean(frame)
    assert len(cleaned) == 2
    assert list(cleaned["timestamp"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-03"]
    names = [record.name for record in log.records]
    assert "sort_chronologically" in names
    assert "drop_duplicates" in names


def test_flag_policy_does_not_fill_missing_prices() -> None:
    frame = generate_synthetic_ohlcv(n_rows=5, seed=1)
    frame.loc[2, "close"] = pd.NA
    log = TransformationLog()
    cleaner = DataCleaner(DataConfig(missing_value_policy="flag"), log)
    cleaned = cleaner.clean(frame)
    assert cleaned["close"].isna().sum() == 1
    assert any(record.name == "missing_policy_flag" for record in log.records)


def test_non_positive_prices_are_errors() -> None:
    frame = generate_synthetic_ohlcv(n_rows=5, seed=2)
    frame.loc[1, "low"] = -0.01
    report = build_quality_report(
        frame,
        symbol="GBPUSD",
        timeframe="1d",
        source="test",
        timezone="UTC",
        expected_frequency="1D",
        abnormal_return_threshold=0.05,
    )
    assert report.n_non_positive_prices >= 1
    assert any(issue.code == "non_positive_prices" for issue in report.issues)
