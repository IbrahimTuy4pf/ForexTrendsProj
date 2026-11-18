"""Shared pytest fixtures for the data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from quant_forex.config.schema import AppConfig
from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.utils.paths import ProjectPaths


@pytest.fixture
def clean_ohlcv() -> pd.DataFrame:
    return generate_synthetic_ohlcv(n_rows=40, seed=7)


@pytest.fixture
def project_paths(tmp_path: Path) -> ProjectPaths:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tmp'\n", encoding="utf-8")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "external").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    (tmp_path / "configs").mkdir()
    return ProjectPaths(root=tmp_path)


@pytest.fixture
def csv_config(tmp_path: Path, clean_ohlcv: pd.DataFrame, project_paths: ProjectPaths) -> AppConfig:
    csv_path = project_paths.data_external / "sample.csv"
    export = clean_ohlcv.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(csv_path, index=False)
    payload = {
        "instrument": {"symbol": "EURUSD", "yfinance_ticker": "EURUSD=X"},
        "timeframe": "1d",
        "data": {
            "source": "csv",
            "csv_path": str(csv_path),
            "timezone": "UTC",
            "missing_value_policy": "flag",
        },
        "storage": {"sqlite_path": str(tmp_path / "data" / "processed" / "market.sqlite")},
        "execution": {"live_trading": False, "paper_trading": False},
    }
    config_path = project_paths.configs / "test.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return AppConfig.model_validate(payload)
