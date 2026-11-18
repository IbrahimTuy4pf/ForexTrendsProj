"""Load processed OHLCV for exploration without mutating source files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_forex.utils.time import ensure_utc_datetime


def load_processed_ohlcv(path: str | Path) -> pd.DataFrame:
    """Read a processed CSV produced by the Phase 1 pipeline (or compatible OHLCV)."""
    frame = pd.read_csv(path, comment="#")
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} has no timestamp column.")
    out = frame.copy()
    out["timestamp"] = ensure_utc_datetime(out["timestamp"], "UTC")
    for column in ("open", "high", "low", "close", "volume"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("timestamp").reset_index(drop=True)


def latest_processed_csv(processed_dir: Path) -> Path | None:
    files = sorted(processed_dir.glob("*_processed.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None
