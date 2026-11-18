"""Pydantic summary of descriptive exploration (not performance attribution)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExplorationSummary(BaseModel):
    symbol: str
    timeframe: str
    n_rows: int
    date_start: str | None
    date_end: str | None
    close_min: float | None
    close_max: float | None
    simple_return_mean: float | None
    simple_return_std: float | None
    simple_return_skew: float | None
    simple_return_excess_kurtosis: float | None
    jarque_bera_statistic: float | None
    jarque_bera_pvalue: float | None
    lag1_return_autocorr: float | None
    return_autocorr: dict[str, float | None] = Field(default_factory=dict)
    n_missing_expected_bars: int
    longest_gap_bars: int
    sessions_applicable: bool
    weekday_mean_return: dict[str, float | None] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
