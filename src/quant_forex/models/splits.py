"""Chronological train / validation / test splits. Never shuffle."""

from __future__ import annotations

import logging

import pandas as pd

from quant_forex.config.schema import SplitConfig

logger = logging.getLogger("quant_forex.models.splits")


def chronological_masks(
    timestamps: pd.Series,
    config: SplitConfig,
) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    ts = pd.to_datetime(timestamps, utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted ascending before splitting.")

    train_end = _parse(config.train_end)
    val_end = _parse(config.validation_end)
    use_dates = (
        train_end is not None
        and val_end is not None
        and ts.min() <= train_end <= ts.max()
        and train_end < val_end
        and ts.max() > train_end
    )
    if use_dates:
        train = ts <= train_end
        valid = (ts > train_end) & (ts <= val_end)
        test = ts > val_end
        if valid.sum() == 0 or test.sum() == 0 or train.sum() == 0:
            use_dates = False
        else:
            return train, valid, test, "configured_dates"

    logger.warning(
        "Configured split dates do not yield three non-empty folds on this sample. "
        "Using chronological fractions instead (not a shuffle)."
    )
    n = len(ts)
    n_train = max(int(n * config.fallback_train_fraction), 1)
    n_val = max(int(n * config.fallback_validation_fraction), 1)
    if n_train + n_val >= n:
        n_train = max(n // 2, 1)
        n_val = max((n - n_train) // 2, 1)
    rank = pd.Series(range(n), index=ts.index)
    train = rank < n_train
    valid = (rank >= n_train) & (rank < n_train + n_val)
    test = rank >= n_train + n_val
    return train, valid, test, "chronological_fractions"


def _parse(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.Timestamp(value, tz="UTC")
