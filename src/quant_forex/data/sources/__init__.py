"""Market data source adapters."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.data.transforms import TransformationLog
from quant_forex.utils.paths import ProjectPaths


class MarketDataSource(Protocol):
    name: str

    def load(self) -> pd.DataFrame:
        """Return a DataFrame with at least timestamp/open/high/low/close."""


def get_source(
    config: AppConfig,
    paths: ProjectPaths,
    transform_log: TransformationLog,
) -> MarketDataSource:
    source = config.data.source
    if source == "csv":
        from quant_forex.data.sources.csv_source import CsvMarketDataSource

        return CsvMarketDataSource(config, paths, transform_log)
    if source == "yfinance":
        from quant_forex.data.sources.yfinance_source import YahooFinanceSource

        return YahooFinanceSource(config, transform_log)
    raise ValueError(f"Unsupported data source: {source}")
