"""Logged data transformations. Raw files are never overwritten."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class TransformationRecord(BaseModel):
    name: str
    description: str
    rows_before: int
    rows_after: int
    parameters: dict[str, Any] = Field(default_factory=dict)
    applied_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


class TransformationLog:
    def __init__(self) -> None:
        self.records: list[TransformationRecord] = []

    def add(
        self,
        name: str,
        description: str,
        rows_before: int,
        rows_after: int,
        parameters: dict[str, Any] | None = None,
    ) -> TransformationRecord:
        record = TransformationRecord(
            name=name,
            description=description,
            rows_before=rows_before,
            rows_after=rows_after,
            parameters=parameters or {},
        )
        self.records.append(record)
        return record

    def as_dicts(self) -> list[dict[str, Any]]:
        return [record.model_dump(mode="json") for record in self.records]
