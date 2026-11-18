"""End-to-end ingestion: load → inspect raw → persist raw → clean → inspect → persist."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.data.quality import build_quality_report, report_to_markdown
from quant_forex.data.sources import get_source
from quant_forex.data.storage import MarketDataStore
from quant_forex.data.transforms import TransformationLog
from quant_forex.data.validation import DataCleaner, flag_abnormal_rows
from quant_forex.utils.paths import ProjectPaths

logger = logging.getLogger("quant_forex.data.pipeline")


@dataclass
class IngestionResult:
    dataset_id: str
    n_raw_rows: int
    n_processed_rows: int
    raw_path: Path
    processed_path: Path
    quality_report_path: Path
    raw_report: object
    processed_report: object


def run_ingestion_pipeline(config: AppConfig, paths: ProjectPaths) -> IngestionResult:
    if config.execution.live_trading:
        raise RuntimeError("Live trading is not available.")

    transform_log = TransformationLog()
    source = get_source(config, paths, transform_log)
    raw = source.load()
    n_raw = len(raw)

    dataset_id = _make_dataset_id(config)
    symbol = config.instrument.symbol
    timeframe = config.timeframe

    raw_report = build_quality_report(
        raw,
        symbol=symbol,
        timeframe=timeframe,
        source=f"{source.name}:raw",
        timezone=config.data.timezone,
        expected_frequency=config.data.expected_frequency,
        abnormal_return_threshold=config.data.abnormal_return_threshold,
        transformations=transform_log.as_dicts(),
    )
    logger.info("Raw quality: rows=%s issues=%s", raw_report.n_rows, len(raw_report.issues))

    paths.data_raw.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

    raw_path = paths.data_raw / f"{dataset_id}_raw.csv"
    processed_path = paths.data_processed / f"{dataset_id}_processed.csv"

    if config.data.persist_raw:
        raw.to_csv(raw_path, index=False)
        logger.info("Wrote raw snapshot to %s (source file is unchanged).", raw_path)

    cleaner = DataCleaner(config.data, transform_log)
    processed = cleaner.clean(raw)
    processed = flag_abnormal_rows(processed)
    processed.insert(0, "symbol", symbol)
    processed.insert(1, "timeframe", timeframe)

    processed_report = build_quality_report(
        processed,
        symbol=symbol,
        timeframe=timeframe,
        source=f"{source.name}:processed",
        timezone="UTC",
        expected_frequency=config.data.expected_frequency,
        abnormal_return_threshold=config.data.abnormal_return_threshold,
        transformations=transform_log.as_dicts(),
    )
    logger.info(
        "Processed quality: rows=%s issues=%s",
        processed_report.n_rows,
        len(processed_report.issues),
    )

    if config.data.persist_processed:
        processed.to_csv(processed_path, index=False)
        logger.info("Wrote processed dataset to %s", processed_path)

    quality_path = paths.reports / f"{dataset_id}_data_quality.md"
    payload = {
        "dataset_id": dataset_id,
        "raw": raw_report.model_dump(mode="json"),
        "processed": processed_report.model_dump(mode="json"),
    }
    quality_json = paths.reports / f"{dataset_id}_data_quality.json"
    quality_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    quality_path.write_text(
        report_to_markdown(raw_report)
        + "\n\n---\n\n"
        + report_to_markdown(processed_report),
        encoding="utf-8",
    )

    store = MarketDataStore(paths.resolve(config.storage.sqlite_path))
    store.record_run(
        dataset_id=dataset_id,
        symbol=symbol,
        timeframe=timeframe,
        source=source.name,
        n_raw_rows=n_raw,
        n_processed_rows=len(processed),
        quality_report_path=str(quality_path),
    )
    store.save_bars(dataset_id, "raw", symbol, timeframe, _bars_only(raw))
    store.save_bars(dataset_id, "processed", symbol, timeframe, processed)

    return IngestionResult(
        dataset_id=dataset_id,
        n_raw_rows=n_raw,
        n_processed_rows=len(processed),
        raw_path=raw_path,
        processed_path=processed_path,
        quality_report_path=quality_path,
        raw_report=raw_report,
        processed_report=processed_report,
    )


def _bars_only(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    return frame.loc[:, columns].copy()


def _make_dataset_id(config: AppConfig) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{config.instrument.symbol}_{config.timeframe}_{stamp}_{short}"
