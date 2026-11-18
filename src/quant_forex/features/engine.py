"""Assemble configured feature groups. Does not assume any feature is useful."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quant_forex.config.schema import FeatureConfig
from quant_forex.features.momentum import add_momentum_features
from quant_forex.features.price import add_price_features
from quant_forex.features.time import add_time_features
from quant_forex.features.trend import add_trend_features
from quant_forex.features.volatility import add_volatility_features

OHLCV_KEEP = ("timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe")


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_columns: list[str]
    groups: list[str]
    n_rows_before: int
    n_rows_after: int
    warmup_notes: list[str] = field(default_factory=list)


def build_features(frame: pd.DataFrame, config: FeatureConfig) -> FeatureBuildResult:
    if "close" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("Feature engine requires timestamp and close columns.")
    keep = [col for col in OHLCV_KEEP if col in frame.columns]
    working = frame.loc[:, keep].sort_values("timestamp").reset_index(drop=True).copy()
    n_before = len(working)
    builders = {
        "price": lambda df: add_price_features(df, config),
        "trend": lambda df: add_trend_features(df, config),
        "momentum": lambda df: add_momentum_features(df, config),
        "volatility": lambda df: add_volatility_features(df, config),
        "time": add_time_features,
    }
    for group in config.groups:
        working = builders[group](working)

    if config.attach_next_return_target:
        working["target_next_simple_return"] = working["close"].pct_change().shift(-1)

    feature_columns = [
        col
        for col in working.columns
        if col not in OHLCV_KEEP and col != "target_next_simple_return"
    ]
    notes = [
        "Warmup NaNs at the start of rolling windows are expected; they are not filled with future data.",
        "Including a feature does not mean it improves a strategy. Use out-of-sample tests later.",
    ]
    n_after = n_before
    if config.drop_warmup_rows and feature_columns:
        before_drop = len(working)
        working = working.dropna(subset=feature_columns).reset_index(drop=True)
        n_after = len(working)
        notes.append(
            f"Dropped {before_drop - n_after} warmup/incomplete rows (drop_warmup_rows=true)."
        )
    else:
        notes.append("Warmup rows kept (drop_warmup_rows=false).")
        n_after = len(working)

    return FeatureBuildResult(
        frame=working,
        feature_columns=feature_columns,
        groups=list(config.groups),
        n_rows_before=n_before,
        n_rows_after=n_after,
        warmup_notes=notes,
    )
