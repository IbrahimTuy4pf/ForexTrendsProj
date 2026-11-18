"""Standardized trading signal. Values come from the strategy and market data."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Direction = Literal["LONG", "SHORT", "FLAT"]


class TradingSignal(BaseModel):
    timestamp: str
    instrument: str
    direction: Direction
    probability: float | None = None
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float | None = None
    risk_percentage: float | None = None
    model_version: str | None = None
    strategy_name: str
    strategy_version: str
    reason: str
    features: dict[str, Any] = Field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        payload = self.model_dump()
        extras = payload.pop("features")
        for key, value in extras.items():
            payload[f"feat_{key}"] = value
        return payload
