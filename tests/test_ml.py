from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_forex.config.schema import AppConfig, FeatureConfig, ModelConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.features.engine import build_features
from quant_forex.models.estimators import build_estimator, predict_proba_up
from quant_forex.models.pipeline import run_ml_pipeline
from quant_forex.models.splits import chronological_masks
from quant_forex.models.target import TARGET_COL, add_next_return_target, feature_matrix_columns


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(compare=["xgboost"])
    with pytest.raises(ValueError, match="Unsupported model"):
        build_estimator("neural_net", ModelConfig(), seed=1)


def test_target_is_not_in_feature_matrix() -> None:
    frame = generate_synthetic_ohlcv(n_rows=80, seed=21)
    built = build_features(frame, FeatureConfig(groups=["price"]))
    labeled = add_next_return_target(built.frame, ModelConfig(horizon=1))
    cols = feature_matrix_columns(labeled)
    assert TARGET_COL not in cols
    assert "close" not in cols
    assert labeled[TARGET_COL].isna().iloc[-1]


def test_chronological_masks_keep_time_order() -> None:
    stamps = pd.Series(pd.date_range("2020-01-01", periods=10, tz="UTC"))
    from quant_forex.config.schema import SplitConfig

    train, val, test, method = chronological_masks(
        stamps,
        SplitConfig(train_end="2015-12-31", validation_end="2016-12-31"),
    )
    assert method == "chronological_fractions"
    assert stamps[train].max() < stamps[val].min()
    assert stamps[val].max() < stamps[test].min()


def test_unsorted_timestamps_are_rejected() -> None:
    stamps = pd.Series(pd.date_range("2020-01-01", periods=5, tz="UTC")[::-1])
    from quant_forex.config.schema import SplitConfig

    with pytest.raises(ValueError, match="sorted ascending"):
        chronological_masks(stamps, SplitConfig())


def test_logistic_scaler_matches_train_mean_only() -> None:
    frame = generate_synthetic_ohlcv(n_rows=120, seed=22)
    built = build_features(frame, FeatureConfig(groups=["price"]))
    labeled = add_next_return_target(built.frame, ModelConfig())
    cols = feature_matrix_columns(labeled)
    usable = labeled.dropna(subset=cols + [TARGET_COL]).reset_index(drop=True)
    from quant_forex.config.schema import SplitConfig

    train, val, test, _ = chronological_masks(usable["timestamp"], SplitConfig())
    x_train = usable.loc[train, cols].to_numpy()
    y_train = usable.loc[train, TARGET_COL].astype(int).to_numpy()
    x_val = usable.loc[val, cols].to_numpy()
    pipe = build_estimator("logistic", ModelConfig(), seed=7)
    pipe.fit(x_train, y_train)
    scaler = pipe.named_steps["scaler"]
    assert int(scaler.n_samples_seen_) == len(x_train)
    assert np.allclose(scaler.mean_, x_train.mean(axis=0))
    assert not np.allclose(scaler.mean_, x_val.mean(axis=0))
    proba = predict_proba_up(pipe, x_val)
    assert proba.shape == (len(x_val),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_ml_pipeline_writes_report_and_keeps_folds_ordered(project_paths) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=180, seed=23)
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    config = AppConfig(
        features=FeatureConfig(groups=["price", "trend"], sma_windows=[10], ema_windows=[10]),
        models=ModelConfig(
            compare=["logistic", "random_forest"],
            n_estimators=20,
            max_depth=2,
            min_samples_leaf=5,
            backtest_test_fold=True,
        ),
    )
    result = run_ml_pipeline(config, project_paths, input_csv=csv_path)
    assert result.report_path.is_file()
    assert result.json_path.is_file()
    assert result.predictions_path.is_file()
    assert result.artifact_path.is_file()
    assert result.selected_model in {"logistic", "random_forest"}
    text = result.report_path.read_text(encoding="utf-8")
    assert "validation ROC-AUC" in text
    preds = pd.read_csv(result.predictions_path)
    assert "probability_up" in preds.columns
    times = pd.to_datetime(preds["timestamp"], utc=True)
    assert times.is_monotonic_increasing
