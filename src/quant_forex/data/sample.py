"""Deterministic synthetic OHLCV for tests and the offline demo.

This is NOT market data. Files generated here must be labeled synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_rows: int = 260,
    start: str = "2024-01-02",
    freq: str = "B",
    seed: int = 42,
    start_price: float = 1.0850,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=n_rows, freq=freq, tz="UTC")
    log_returns = rng.normal(loc=0.0, scale=0.004, size=n_rows)
    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0.0001, 0.0020, size=n_rows)
    low = np.minimum(open_, close) - rng.uniform(0.0001, 0.0020, size=n_rows)
    volume = rng.integers(1000, 5000, size=n_rows).astype(float)
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
