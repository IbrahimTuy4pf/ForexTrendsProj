"""Explicit cleaning steps. Every mutation is copied and logged."""

from __future__ import annotations

import logging

import pandas as pd

from quant_forex.config.schema import DataConfig
from quant_forex.data.quality import inspect_ohlc_consistency
from quant_forex.data.transforms import TransformationLog
from quant_forex.utils.time import ensure_utc_datetime

logger = logging.getLogger("quant_forex.data.validation")


class DataCleaner:
    """Apply configured cleaning. Raw data is never overwritten."""

    def __init__(self, data_config: DataConfig, transform_log: TransformationLog) -> None:
        self.data_config = data_config
        self.log = transform_log

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        current = frame.copy()
        current = self._normalize_timestamps(current)
        current = self._coerce_numeric(current)
        if self.data_config.sort_chronologically:
            current = self._sort(current)
        if self.data_config.drop_duplicates:
            current = self._drop_duplicates(current)
        current = self._handle_missing(current)
        if self.data_config.resample_rule:
            current = self._resample(current)
        return current.reset_index(drop=True)

    def _normalize_timestamps(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in frame.columns:
            return frame
        before = len(frame)
        out = frame.copy()
        out["timestamp"] = ensure_utc_datetime(out["timestamp"], self.data_config.timezone)
        dropped = out["timestamp"].isna().sum()
        if dropped:
            logger.warning("Dropping %s rows with unparseable timestamps.", int(dropped))
            out = out.loc[out["timestamp"].notna()].copy()
        self.log.add(
            "normalize_timestamps",
            "Parse timestamps and convert to UTC. Unparseable rows are dropped, not guessed.",
            before,
            len(out),
            {"timezone": self.data_config.timezone, "dropped_unparsed": int(dropped)},
        )
        return out

    def _coerce_numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in ("open", "high", "low", "close", "volume"):
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce")
        if "volume" not in out.columns:
            out["volume"] = pd.NA
            self.log.add(
                "add_empty_volume",
                "Volume column was missing; added as NA. Values were not invented.",
                len(out),
                len(out),
                {},
            )
        self.log.add(
            "coerce_numeric",
            "Coerce OHLC(V) columns to numeric; invalid numbers become NA.",
            len(out),
            len(out),
            {},
        )
        return out

    def _sort(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in frame.columns:
            return frame
        before = len(frame)
        out = frame.sort_values("timestamp").copy()
        self.log.add(
            "sort_chronologically",
            "Sort bars by timestamp ascending.",
            before,
            len(out),
            {},
        )
        return out

    def _drop_duplicates(self, frame: pd.DataFrame) -> pd.DataFrame:
        before = len(frame)
        out = frame.drop_duplicates().copy()
        if "timestamp" in out.columns:
            n_ts_dups = int(out["timestamp"].duplicated().sum())
            if n_ts_dups:
                logger.warning(
                    "Dropping %s extra rows that share a timestamp (keeping first).",
                    n_ts_dups,
                )
                out = out.drop_duplicates(subset=["timestamp"], keep="first")
        self.log.add(
            "drop_duplicates",
            "Drop exact duplicate rows, then extra rows that share a timestamp (keep first).",
            before,
            len(out),
            {"rows_removed": before - len(out)},
        )
        return out

    def _handle_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        policy = self.data_config.missing_value_policy
        price_cols = [c for c in ("open", "high", "low", "close") if c in frame.columns]
        before = len(frame)
        out = frame.copy()
        if policy == "flag":
            self.log.add(
                "missing_policy_flag",
                "Missing values left in place. No prices were filled or interpolated.",
                before,
                len(out),
                {"policy": policy},
            )
            return out
        if policy == "drop":
            out = out.dropna(subset=price_cols)
            self.log.add(
                "missing_policy_drop",
                "Dropped rows with missing OHLC values.",
                before,
                len(out),
                {"policy": policy},
            )
            return out
        if policy == "ffill":
            out[price_cols] = out[price_cols].ffill()
            self.log.add(
                "missing_policy_ffill",
                "Forward-filled missing OHLC only. No interpolation and no backfill.",
                before,
                len(out),
                {"policy": policy},
            )
            return out
        raise ValueError(f"Unknown missing_value_policy: {policy}")

    def _resample(self, frame: pd.DataFrame) -> pd.DataFrame:
        rule = self.data_config.resample_rule
        if not rule or "timestamp" not in frame.columns:
            return frame
        before = len(frame)
        indexed = frame.set_index("timestamp").sort_index()
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in indexed.columns:
            agg["volume"] = "sum"
        resampled = indexed.resample(rule).agg(agg).dropna(how="all").reset_index()
        self.log.add(
            "resample",
            "Resample OHLCV with standard candle aggregation.",
            before,
            len(resampled),
            {"rule": rule},
        )
        return resampled


def flag_abnormal_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Add boolean diagnostic columns without removing rows."""
    out = frame.copy()
    if all(c in out.columns for c in ("open", "high", "low", "close")):
        out["flag_inconsistent_ohlc"] = inspect_ohlc_consistency(out)
        prices = out[["open", "high", "low", "close"]]
        out["flag_non_positive_price"] = (prices <= 0).any(axis=1)
    return out
