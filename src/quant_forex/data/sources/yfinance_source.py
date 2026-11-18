"""Download daily FX from Yahoo Finance via yfinance.

Yahoo FX is convenient for education. It is not tick-level broker data, can have
gaps around weekends/holidays, and should not be treated as a production feed.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.data.schema import standardize_ohlcv_columns
from quant_forex.data.transforms import TransformationLog

logger = logging.getLogger("quant_forex.data.yfinance")


class YahooFinanceSource:
    name = "yfinance"

    def __init__(self, config: AppConfig, transform_log: TransformationLog) -> None:
        self.config = config
        self.log = transform_log

    def load(self) -> pd.DataFrame:
        import yfinance as yf

        ticker = self.config.instrument.resolved_yfinance_ticker()
        period = self.config.data.yfinance_period
        interval = self.config.data.yfinance_interval
        logger.info("Downloading %s period=%s interval=%s", ticker, period, interval)
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(
                f"yfinance returned no rows for {ticker}. Check the ticker and your network."
            )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [
                col[0] if isinstance(col, tuple) else col for col in raw.columns
            ]
        raw = raw.reset_index()
        frame, mapping = standardize_ohlcv_columns(raw)
        self.log.add(
            "yfinance_download",
            "Download public Yahoo Finance FX history. This is not broker-quality tick data.",
            0,
            len(frame),
            {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "column_mapping": mapping,
                "auto_adjust": False,
            },
        )
        return frame
