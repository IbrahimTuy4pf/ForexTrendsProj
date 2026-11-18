"""Binary target: next-horizon close return exceeds a threshold.

The label uses future closes. Those closes must never appear in X.
"""

from __future__ import annotations

import pandas as pd

from quant_forex.config.schema import ModelConfig
from quant_forex.features.engine import OHLCV_KEEP

TARGET_COL = "target_up"


def add_next_return_target(frame: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    if "close" not in frame.columns:
        raise ValueError("close is required to build the target.")
    out = frame.copy()
    future = out["close"].shift(-config.horizon)
    fwd = future / out["close"] - 1.0
    out[TARGET_COL] = (fwd > config.threshold).astype("float64")
    out.loc[future.isna(), TARGET_COL] = pd.NA
    return out


def feature_matrix_columns(frame: pd.DataFrame) -> list[str]:
    banned = set(OHLCV_KEEP) | {TARGET_COL, "target_next_simple_return"}
    cols: list[str] = []
    for column in frame.columns:
        if column in banned:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        cols.append(column)
    return cols
