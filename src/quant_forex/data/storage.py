"""SQLite persistence for OHLCV bars and dataset lineage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class DatasetRun(Base):
    __tablename__ = "dataset_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    n_raw_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_processed_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_report_path: Mapped[str] = mapped_column(String(512), nullable=False)


class OhlcvBar(Base):
    __tablename__ = "ohlcv_bars"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "kind",
            "timestamp",
            name="uq_bar_dataset_kind_ts",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)


class MarketDataStore:
    def __init__(self, sqlite_path: Path) -> None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{sqlite_path.as_posix()}"
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)

    def record_run(
        self,
        dataset_id: str,
        symbol: str,
        timeframe: str,
        source: str,
        n_raw_rows: int,
        n_processed_rows: int,
        quality_report_path: str,
    ) -> None:
        run = DatasetRun(
            id=dataset_id,
            symbol=symbol,
            timeframe=timeframe,
            source=source,
            created_at_utc=datetime.now(UTC),
            n_raw_rows=n_raw_rows,
            n_processed_rows=n_processed_rows,
            quality_report_path=quality_report_path,
        )
        with Session(self.engine) as session:
            session.add(run)
            session.commit()

    def save_bars(
        self,
        dataset_id: str,
        kind: str,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
    ) -> None:
        if frame.empty:
            return
        rows: list[OhlcvBar] = []
        for record in frame.to_dict(orient="records"):
            ts = _as_utc_datetime(record.get("timestamp"))
            if ts is None:
                continue
            rows.append(
                OhlcvBar(
                    dataset_id=dataset_id,
                    kind=kind,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=_optional_float(record.get("open")),
                    high=_optional_float(record.get("high")),
                    low=_optional_float(record.get("low")),
                    close=_optional_float(record.get("close")),
                    volume=_optional_float(record.get("volume")),
                )
            )
        with Session(self.engine) as session:
            session.add_all(rows)
            session.commit()

    def load_bars(self, dataset_id: str, kind: str = "processed") -> pd.DataFrame:
        with Session(self.engine) as session:
            stmt = (
                select(OhlcvBar)
                .where(OhlcvBar.dataset_id == dataset_id, OhlcvBar.kind == kind)
                .order_by(OhlcvBar.timestamp)
            )
            bars = session.scalars(stmt).all()
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe,
                }
                for bar in bars
            ]
        )


def _as_utc_datetime(value: object) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    stamp = pd.to_datetime(value, utc=True)
    if pd.isna(stamp):
        return None
    return stamp.to_pydatetime()


def _optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)
