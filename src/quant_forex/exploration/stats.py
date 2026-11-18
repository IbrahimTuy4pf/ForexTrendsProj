"""Descriptive statistics. These are not strategy performance metrics."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from quant_forex.exploration.gaps import gap_table
from quant_forex.exploration.summary import ExplorationSummary


def rolling_return_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Causal rolling std of returns (includes the current bar, no future bars)."""
    return returns.rolling(window=window, min_periods=window).std()


def return_autocorrelations(returns: pd.Series, lags: int) -> dict[str, float | None]:
    clean = returns.dropna()
    out: dict[str, float | None] = {}
    for lag in range(1, lags + 1):
        if len(clean) <= lag:
            out[f"lag_{lag}"] = None
        else:
            out[f"lag_{lag}"] = float(clean.autocorr(lag=lag))
    return out


def weekday_mean_returns(frame: pd.DataFrame) -> dict[str, float | None]:
    if "simple_return" not in frame.columns or "day_of_week" not in frame.columns:
        return {}
    grouped = frame.groupby("day_of_week", observed=True)["simple_return"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return {day: (float(grouped[day]) if day in grouped.index else None) for day in order if day in grouped.index}


def build_exploration_summary(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    expected_frequency: str | None,
    autocorr_lags: int,
    source_note: str | None = None,
) -> ExplorationSummary:
    returns = frame["simple_return"].dropna() if "simple_return" in frame.columns else pd.Series(dtype=float)
    close = frame["close"].dropna() if "close" in frame.columns else pd.Series(dtype=float)
    stamps = pd.to_datetime(frame["timestamp"], utc=True) if "timestamp" in frame.columns else pd.Series(dtype="datetime64[ns, UTC]")

    skew = kurt = jb_stat = jb_p = None
    if len(returns) >= 8:
        skew = float(stats.skew(returns, bias=False))
        kurt = float(stats.kurtosis(returns, fisher=True, bias=False))
        jb_stat, jb_p = (float(x) for x in stats.jarque_bera(returns))

    ac = return_autocorrelations(returns, autocorr_lags)
    gaps = gap_table(stamps, expected_frequency)
    longest = int(gaps["n_missing_bars"].max()) if not gaps.empty else 0
    sessions_applicable = bool(frame["session_applicable"].any()) if "session_applicable" in frame.columns else False

    caveats = [
        "Descriptive statistics are not a backtest and do not include spread, slippage, or overnight financing.",
        "A non-normal return distribution is expected in FX; it is not evidence of an edge.",
        "Autocorrelation near zero is consistent with (but does not prove) weak-form efficiency.",
    ]
    if source_note:
        caveats.insert(0, source_note)
    if not sessions_applicable:
        caveats.append(
            "Session flags are not applicable: timestamps do not vary by hour "
            "(typical of daily bars). Use intraday data to study London/New York sessions."
        )

    return ExplorationSummary(
        symbol=symbol,
        timeframe=timeframe,
        n_rows=int(len(frame)),
        date_start=stamps.min().isoformat() if not stamps.empty else None,
        date_end=stamps.max().isoformat() if not stamps.empty else None,
        close_min=float(close.min()) if not close.empty else None,
        close_max=float(close.max()) if not close.empty else None,
        simple_return_mean=float(returns.mean()) if not returns.empty else None,
        simple_return_std=float(returns.std(ddof=1)) if len(returns) > 1 else None,
        simple_return_skew=skew,
        simple_return_excess_kurtosis=kurt,
        jarque_bera_statistic=jb_stat,
        jarque_bera_pvalue=jb_p,
        lag1_return_autocorr=ac.get("lag_1"),
        return_autocorr=ac,
        n_missing_expected_bars=int(gaps["n_missing_bars"].sum()) if not gaps.empty else 0,
        longest_gap_bars=longest,
        sessions_applicable=sessions_applicable,
        weekday_mean_return=weekday_mean_returns(frame),
        caveats=caveats,
        extra={"n_return_observations": int(len(returns))},
    )


def cumulative_log_return(frame: pd.DataFrame) -> pd.Series:
    if "log_return" not in frame.columns:
        raise ValueError("log_return column required")
    return frame["log_return"].fillna(0.0).cumsum()
