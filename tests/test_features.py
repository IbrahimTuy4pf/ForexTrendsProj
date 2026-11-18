from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_forex.config.schema import FeatureConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.features.engine import build_features
from quant_forex.features.importance import next_bar_associations
from quant_forex.features.pipeline import run_feature_pipeline
from quant_forex.features.trend import add_trend_features


def test_unknown_feature_group_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureConfig(groups=["deep_learning"])


def test_sma_matches_causal_window() -> None:
    frame = generate_synthetic_ohlcv(n_rows=30, seed=11)
    config = FeatureConfig(groups=["trend"], sma_windows=[5], ema_windows=[5])
    built = build_features(frame, config).frame
    idx = 10
    expected = frame["close"].iloc[idx - 4 : idx + 1].mean()
    assert built.loc[idx, "sma_5"] == pytest.approx(expected)


def test_future_close_does_not_change_earlier_features() -> None:
    frame = generate_synthetic_ohlcv(n_rows=80, seed=12)
    config = FeatureConfig()
    original = build_features(frame, config).frame
    mutated = frame.copy()
    mutated.loc[mutated.index[-1], "close"] = mutated.loc[mutated.index[-1], "close"] * 1.5
    mutated.loc[mutated.index[-1], "high"] = max(
        mutated.loc[mutated.index[-1], "high"], mutated.loc[mutated.index[-1], "close"]
    )
    redone = build_features(mutated, config).frame
    compare_cols = [
        col
        for col in original.columns
        if col not in {"close", "high", "open", "low", "volume"}
    ]
    pd.testing.assert_frame_equal(
        original.iloc[:-1][compare_cols].reset_index(drop=True),
        redone.iloc[:-1][compare_cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_disabled_trend_group_omits_sma() -> None:
    frame = generate_synthetic_ohlcv(n_rows=40, seed=13)
    built = build_features(frame, FeatureConfig(groups=["price", "time"]))
    assert all(not col.startswith("sma_") for col in built.feature_columns)
    assert "ret_1" in built.feature_columns
    assert "hour_utc" in built.feature_columns


def test_vol_regime_is_not_full_sample_quantile() -> None:
    frame = generate_synthetic_ohlcv(n_rows=100, seed=14)
    config = FeatureConfig(groups=["volatility"], vol_window=5, vol_regime_window=10)
    built = build_features(frame, config).frame
    idx = 40
    vol = built["vol_5"]
    median = vol.rolling(window=10, min_periods=10).median()
    expected = float(vol.iloc[idx] > median.iloc[idx])
    assert built.loc[idx, "vol_regime_high"] == pytest.approx(expected)


def test_next_bar_association_does_not_put_target_in_features() -> None:
    frame = generate_synthetic_ohlcv(n_rows=50, seed=15)
    built = build_features(frame, FeatureConfig(groups=["price"], rolling_return_windows=[5]))
    table = next_bar_associations(built.frame, built.feature_columns)
    assert "target_next_simple_return" not in built.feature_columns
    assert not built.frame.columns.str.contains("target").any()
    assert set(table["feature"]) <= set(built.feature_columns)


def test_optional_target_column_is_shifted_lead() -> None:
    frame = generate_synthetic_ohlcv(n_rows=12, seed=16)
    built = build_features(frame, FeatureConfig(groups=["price"], attach_next_return_target=True))
    assert pd.isna(built.frame["target_next_simple_return"].iloc[-1])
    expected = frame["close"].iloc[3] / frame["close"].iloc[2] - 1.0
    assert built.frame.loc[2, "target_next_simple_return"] == pytest.approx(expected)


def test_drop_warmup_removes_indicator_nan_rows() -> None:
    frame = generate_synthetic_ohlcv(n_rows=40, seed=17)
    kept = build_features(frame, FeatureConfig(groups=["trend"], sma_windows=[10], drop_warmup_rows=False))
    dropped = build_features(frame, FeatureConfig(groups=["trend"], sma_windows=[10], drop_warmup_rows=True))
    assert dropped.n_rows_after < kept.n_rows_after
    assert dropped.frame["sma_10"].notna().all()


def test_feature_pipeline_writes_csv_and_report(project_paths) -> None:
    csv_path = project_paths.data_processed / "GBPUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=90, seed=18)
    frame.insert(0, "symbol", "GBPUSD")
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    from quant_forex.config.schema import AppConfig

    config = AppConfig(features=FeatureConfig(groups=["price", "trend"], sma_windows=[10]))
    result = run_feature_pipeline(config, project_paths, input_csv=csv_path)
    assert result.output_csv.is_file()
    assert result.report_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "No feature is assumed useful" in text
    out = pd.read_csv(result.output_csv)
    assert "sma_10" in out.columns
    assert out["symbol"].iloc[0] == "GBPUSD"


def test_ema_adjust_false_is_recursive() -> None:
    close = pd.Series(np.linspace(1.0, 1.1, 20))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, tz="UTC"),
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": 1.0,
        }
    )
    config = FeatureConfig(groups=["trend"], sma_windows=[5], ema_windows=[5])
    built = add_trend_features(frame, config)
    expected = close.ewm(span=5, adjust=False, min_periods=5).mean()
    pd.testing.assert_series_equal(built["ema_5"], expected, check_names=False)
