"""Load OHLCV from a local CSV file."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.data.schema import standardize_ohlcv_columns
from quant_forex.data.transforms import TransformationLog
from quant_forex.utils.paths import ProjectPaths

logger = logging.getLogger("quant_forex.data.csv")


class CsvMarketDataSource:
    name = "csv"

    def __init__(
        self,
        config: AppConfig,
        paths: ProjectPaths,
        transform_log: TransformationLog,
    ) -> None:
        self.config = config
        self.paths = paths
        self.log = transform_log

    def load(self) -> pd.DataFrame:
        csv_path = self.config.data.csv_path
        if not csv_path:
            raise ValueError("data.csv_path is required when source=csv.")
        resolved = Path(csv_path)
        if not resolved.is_file():
            resolved = self.paths.resolve(csv_path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}. See README for where to obtain FX data."
            )

        logger.info("Loading CSV from %s", resolved)
        raw = pd.read_csv(resolved, comment="#", skip_blank_lines=True)
        n_before = len(raw)
        frame, mapping = standardize_ohlcv_columns(raw)
        self.log.add(
            "csv_column_standardization",
            "Rename columns to canonical OHLCV names. Values are unchanged.",
            n_before,
            len(frame),
            {"path": str(resolved), "column_mapping": mapping},
        )
        return frame
