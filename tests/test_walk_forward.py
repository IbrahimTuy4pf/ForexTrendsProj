from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from quant_forex.config.schema import AppConfig, FeatureConfig, ModelConfig, WalkForwardConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.evaluation.folds import generate_walk_forward_folds
from quant_forex.evaluation.pipeline import run_walk_forward
from quant_forex.features.engine import build_features
from quant_forex.models.target import TARGET_COL, add_next_return_target, feature_matrix_columns


def test_validation_cannot_consume_entire_train_window() -> None:
    with pytest.raises(ValidationError):
        WalkForwardConfig(train_bars=20, validation_bars=20)


def test_folds_are_chronological_and_purged() -> None:
    cfg = WalkForwardConfig(
        mode="expanding",
        train_bars=40,
        validation_bars=10,
        test_bars=10,
        step_bars=10,
        embargo_bars=2,
    )
    horizon = 3
    folds = generate_walk_forward_folds(100, cfg, horizon=horizon)
    assert len(folds) >= 2
    for fold in folds:
        fold.assert_order()
        assert fold.fit_start == 0
        assert fold.test_start - fold.val_end == horizon + cfg.embargo_bars
        assert fold.fit_end == fold.val_start
        last_train = fold.val_end - 1
        last_label_price_index = last_train + horizon
        assert last_label_price_index < fold.test_start


def test_rolling_caps_the_fit_window() -> None:
    cfg = WalkForwardConfig(
        mode="rolling",
        train_bars=40,
        validation_bars=10,
        test_bars=10,
        step_bars=10,
        embargo_bars=0,
    )
    folds = generate_walk_forward_folds(120, cfg, horizon=1)
    later = [item for item in folds if item.fit_start > 0]
    assert later
    for fold in later:
        assert fold.val_end - fold.fit_start == cfg.train_bars
        assert fold.fit_end - fold.fit_start == cfg.train_bars - cfg.validation_bars


def test_non_overlapping_tests_when_step_equals_test() -> None:
    cfg = WalkForwardConfig(train_bars=30, validation_bars=8, test_bars=8, step_bars=8)
    folds = generate_walk_forward_folds(80, cfg, horizon=1)
    starts = [item.test_start for item in folds]
    ends = [item.test_end for item in folds]
    for prev_end, nxt in zip(ends, starts[1:], strict=False):
        assert nxt >= prev_end


def test_walk_forward_pipeline_refits_without_using_future(project_paths) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=200, seed=31)
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    config = AppConfig(
        features=FeatureConfig(groups=["price"], rolling_return_windows=[5]),
        models=ModelConfig(
            compare=["logistic"],
            min_samples_leaf=5,
            backtest_test_fold=False,
        ),
        walk_forward=WalkForwardConfig(
            mode="expanding",
            train_bars=50,
            validation_bars=10,
            test_bars=10,
            step_bars=10,
            embargo_bars=0,
            backtest_oos=True,
        ),
    )
    result = run_walk_forward(config, project_paths, input_csv=csv_path)
    assert result.n_folds >= 2
    assert result.report_path.is_file()
    assert result.predictions_path.is_file()
    preds = pd.read_csv(result.predictions_path)
    times = pd.to_datetime(preds["timestamp"], utc=True)
    assert times.is_monotonic_increasing
    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    for fold in data["folds"]:
        assert pd.Timestamp(fold["fit_end"]) < pd.Timestamp(fold["val_start"])
        assert pd.Timestamp(fold["val_end"]) < pd.Timestamp(fold["test_start"])
    built = build_features(frame, config.features)
    labeled = add_next_return_target(built.frame, config.models)
    assert TARGET_COL not in feature_matrix_columns(labeled)
    text = result.report_path.read_text(encoding="utf-8")
    assert "Walk-forward" in text
    assert data["mean_test_roc_auc"] is None or isinstance(data["mean_test_roc_auc"], float)
