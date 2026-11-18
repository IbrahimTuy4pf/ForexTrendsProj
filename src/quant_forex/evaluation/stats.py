"""Block bootstrap and permutation tests on already-costed results."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def bar_returns(equity: pd.Series) -> np.ndarray:
    rets = pd.Series(equity).astype(float).pct_change().dropna().to_numpy(dtype=float)
    return rets[np.isfinite(rets)]


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float | None:
    if len(returns) < 2:
        return None
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return None
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float] | None:
    clean = np.asarray(samples, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return None
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(clean, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def moving_block_bootstrap(
    returns: np.ndarray,
    *,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    n = len(returns)
    if n < block_size or n < 5:
        return np.array([])
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_size))
    out = np.empty((n_bootstrap, n), dtype=float)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        pieces = [np.take(returns, (np.arange(block_size) + s) % n) for s in starts]
        out[i] = np.concatenate(pieces)[:n]
    return out


def bootstrap_summary(
    returns: np.ndarray,
    *,
    n_bootstrap: int,
    block_size: int,
    confidence_level: float,
    seed: int,
    periods_per_year: int,
) -> dict[str, object]:
    observed_mean = float(np.mean(returns)) if len(returns) else None
    observed_sharpe = sharpe_ratio(returns, periods_per_year)
    samples = moving_block_bootstrap(
        returns, n_bootstrap=n_bootstrap, block_size=block_size, seed=seed
    )
    if len(samples) == 0:
        return {
            "n": int(len(returns)),
            "mean_return": observed_mean,
            "sharpe": observed_sharpe,
            "mean_ci": None,
            "sharpe_ci": None,
            "p_mean_le_zero": None,
        }
    boot_means = samples.mean(axis=1)
    boot_sharpes = np.array(
        [sharpe_ratio(row, periods_per_year) or np.nan for row in samples], dtype=float
    )
    p_mean = float(np.mean(boot_means <= 0.0)) if observed_mean is not None else None
    return {
        "n": int(len(returns)),
        "mean_return": observed_mean,
        "sharpe": observed_sharpe,
        "mean_ci": percentile_ci(boot_means, confidence_level),
        "sharpe_ci": percentile_ci(boot_sharpes, confidence_level),
        "p_mean_le_zero": p_mean,
        "n_bootstrap": int(len(samples)),
        "block_size": block_size,
        "note": "Moving-block bootstrap of equity returns (not a full path-dependent re-backtest).",
    }


def calendar_tables(equity_curve: pd.DataFrame) -> dict[str, dict[str, float]]:
    if equity_curve.empty or "timestamp" not in equity_curve.columns:
        return {"monthly": {}, "annual": {}}
    frame = equity_curve.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").set_index("timestamp")
    equity = frame["equity"].astype(float)
    monthly = equity.resample("ME").last().pct_change().dropna()
    annual = equity.resample("YE").last().pct_change().dropna()
    return {
        "monthly": {idx.strftime("%Y-%m"): float(val) for idx, val in monthly.items()},
        "annual": {idx.strftime("%Y"): float(val) for idx, val in annual.items()},
    }
