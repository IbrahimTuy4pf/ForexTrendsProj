"""Naive next-bar diagnostics. This is NOT the Phase 5 backtest engine."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel


class NaiveDiagnostics(BaseModel):
    strategy: str
    n_bars: int
    n_long: int
    n_short: int
    n_flat: int
    n_entries: int
    naive_sum_next_return: float | None
    naive_hit_rate: float | None
    n_active_next_bars: int


def naive_next_bar_diagnostics(
    frame: pd.DataFrame,
    positions: pd.Series,
    strategy: str,
    n_entries: int,
) -> NaiveDiagnostics:
    next_ret = frame["close"].pct_change().shift(-1)
    active = positions.fillna(0.0) != 0.0
    strat_ret = positions.fillna(0.0) * next_ret
    usable = strat_ret.loc[active & strat_ret.notna()]
    hit = float((usable > 0).mean()) if len(usable) else None
    total = float(strat_ret.dropna().sum()) if strat_ret.notna().any() else None
    pos = positions.fillna(0.0)
    return NaiveDiagnostics(
        strategy=strategy,
        n_bars=int(len(frame)),
        n_long=int((pos > 0).sum()),
        n_short=int((pos < 0).sum()),
        n_flat=int((pos == 0).sum()),
        n_entries=n_entries,
        naive_sum_next_return=total,
        naive_hit_rate=hit,
        n_active_next_bars=int(len(usable)),
    )
