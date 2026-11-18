from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_forex.config.schema import AppConfig, DiagnosticsConfig, FeatureConfig, ModelConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.evaluation.diagnose import run_diagnostics
from quant_forex.evaluation.flags import TOO_GOOD_MESSAGE, auc_gap_flags, performance_flags
from quant_forex.evaluation.stats import bar_returns, bootstrap_summary, moving_block_bootstrap, sharpe_ratio


def test_too_good_flag_matches_brief() -> None:
    flags = performance_flags(
        total_return=0.80,
        max_drawdown=0.01,
        sharpe=4.0,
        win_rate=0.5,
        n_trades=40,
        too_good_total_return=0.50,
        too_good_max_drawdown=0.03,
        too_good_sharpe=2.5,
    )
    assert TOO_GOOD_MESSAGE in flags


def test_quiet_results_are_not_flagged() -> None:
    flags = performance_flags(
        total_return=0.02,
        max_drawdown=0.12,
        sharpe=0.1,
        win_rate=0.48,
        n_trades=40,
        too_good_total_return=0.50,
        too_good_max_drawdown=0.03,
        too_good_sharpe=2.5,
    )
    assert flags == []


def test_auc_gap_is_flagged() -> None:
    flags = auc_gap_flags(0.92, 0.51, 0.15)
    assert flags
    assert not auc_gap_flags(0.55, 0.52, 0.15)


def test_block_bootstrap_preserves_length_and_ci_covers_mean() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=200)
    samples = moving_block_bootstrap(returns, n_bootstrap=80, block_size=10, seed=1)
    assert samples.shape == (80, 200)
    summary = bootstrap_summary(
        returns,
        n_bootstrap=80,
        block_size=10,
        confidence_level=0.95,
        seed=1,
        periods_per_year=252,
    )
    lo, hi = summary["mean_ci"]
    assert lo < float(np.mean(returns)) < hi
    assert summary["sharpe"] == sharpe_ratio(returns, 252)


def test_invalid_diagnostics_config() -> None:
    with pytest.raises(ValidationError):
        DiagnosticsConfig(confidence_level=1.5)


def test_diagnostics_pipeline_writes_report(project_paths) -> None:
    csv_path = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=90, seed=41)
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    config = AppConfig(
        features=FeatureConfig(groups=["price"]),
        models=ModelConfig(compare=["logistic"]),
        diagnostics=DiagnosticsConfig(
            n_bootstrap=24,
            n_permutations=6,
            block_size=5,
            ma_fast_offsets=[-1, 0, 1],
            ma_slow_offsets=[-2, 0, 2],
            momentum_lookback_offsets=[0],
        ),
    )
    config.strategies.mean_reversion.enabled = False
    config.strategies.breakout.enabled = False
    config.strategies.momentum.enabled = False
    ml_payload = {
        "selected_model": "logistic",
        "metrics": {
            "logistic": {
                "train": {"roc_auc": 0.99},
                "test": {"roc_auc": 0.50},
            }
        },
        "feature_importance": {"logistic": [{"feature": "ret_1", "importance": 1.0}]},
    }
    (project_paths.reports / "EURUSD_1d_demo_ml.json").write_text(
        json.dumps(ml_payload), encoding="utf-8"
    )
    result = run_diagnostics(config, project_paths, input_csv=csv_path)
    assert result.report_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "Diagnostics report" in text
    data = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert data["strategies"]
    assert any("degraded out of sample" in item for item in data["flags"])
    equity = pd.Series(np.linspace(10_000, 10_200, 20))
    assert len(bar_returns(equity)) == 19
