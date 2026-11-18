"""Canonical OHLCV column names and light DataFrame contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

COLUMN_ALIASES: dict[str, str] = {
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "dt": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
    "adj close": "close",
    "adj_close": "close",
}


def standardize_ohlcv_columns(
    frame: pd.DataFrame,
    extra_aliases: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns to the canonical OHLCV schema.

    Returns the renamed frame and a mapping of original -> canonical names.
    Unknown columns are preserved. Missing required OHLC columns are not invented.
    """
    aliases = {**COLUMN_ALIASES, **(extra_aliases or {})}
    mapping: dict[str, str] = {}
    new_names: list[str] = []
    used_canonical: set[str] = set()

    for original in frame.columns:
        key = str(original).strip().lower()
        canonical = aliases.get(key, key)
        if canonical in OHLCV_COLUMNS and canonical not in used_canonical:
            mapping[str(original)] = canonical
            new_names.append(canonical)
            used_canonical.add(canonical)
        else:
            new_names.append(str(original))

    renamed = frame.copy()
    renamed.columns = new_names
    if "timestamp" not in renamed.columns and renamed.index.name in {
        "date",
        "datetime",
        "timestamp",
        "time",
    }:
        renamed = renamed.reset_index()
        if renamed.columns[0] != "timestamp":
            renamed = renamed.rename(columns={renamed.columns[0]: "timestamp"})
            mapping[str(frame.index.name)] = "timestamp"
    elif "timestamp" not in renamed.columns and not isinstance(renamed.index, pd.RangeIndex):
        if isinstance(renamed.index, pd.DatetimeIndex):
            renamed = renamed.reset_index()
            first = renamed.columns[0]
            renamed = renamed.rename(columns={first: "timestamp"})
            mapping["index"] = "timestamp"

    return renamed, mapping
