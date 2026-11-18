from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_forex.data.pipeline import run_ingestion_pipeline
from quant_forex.data.storage import MarketDataStore


def test_pipeline_writes_raw_processed_sqlite_and_quality_report(
    csv_config, project_paths
) -> None:
    source_csv = Path(csv_config.data.csv_path)
    original = source_csv.read_text(encoding="utf-8")

    result = run_ingestion_pipeline(csv_config, project_paths)

    assert result.n_raw_rows == 40
    assert result.n_processed_rows == 40
    assert result.raw_path.is_file()
    assert result.processed_path.is_file()
    assert result.quality_report_path.is_file()
    markdown = result.quality_report_path.read_text(encoding="utf-8")
    assert "Data quality report" in markdown
    assert "EURUSD" in markdown
    assert source_csv.read_text(encoding="utf-8") == original

    processed = pd.read_csv(result.processed_path)
    assert "symbol" in processed.columns
    assert set(processed["symbol"].unique()) == {"EURUSD"}

    store = MarketDataStore(project_paths.resolve(csv_config.storage.sqlite_path))
    bars = store.load_bars(result.dataset_id, kind="processed")
    assert len(bars) == 40
    assert list(bars["timestamp"]) == sorted(bars["timestamp"].tolist())


def test_pipeline_uses_configured_symbol(csv_config, project_paths, clean_ohlcv) -> None:
    csv_config.instrument.symbol = "AUDUSD"
    csv_config.instrument.base_currency = "AUD"
    csv_config.instrument.quote_currency = "USD"
    result = run_ingestion_pipeline(csv_config, project_paths)
    processed = pd.read_csv(result.processed_path)
    assert processed["symbol"].iloc[0] == "AUDUSD"
    assert "AUDUSD" in result.dataset_id
