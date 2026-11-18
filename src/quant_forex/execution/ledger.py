"""Local SQLite paper ledger. This is not a broker account."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class PaperSession(Base):
    __tablename__ = "paper_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(32), nullable=False, default="paper_simulator")
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_bar_utc: Mapped[str] = mapped_column(String(64), nullable=False)
    n_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    units: Mapped[float] = mapped_column(Float, nullable=False)
    pending_json: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False)


class PaperFill(Base):
    __tablename__ = "paper_fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    direction: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_time: Mapped[str] = mapped_column(String(64), nullable=False)
    exit_time: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    units: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False)
    costs: Mapped[float] = mapped_column(Float, nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(32), nullable=False, default="paper_simulator")


class PaperLedger:
    def __init__(self, sqlite_path: Path) -> None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = sqlite_path
        self.engine = create_engine(f"sqlite:///{sqlite_path.as_posix()}", future=True)
        Base.metadata.create_all(self.engine)

    def replace_session(
        self,
        *,
        symbol: str,
        strategy: str,
        last_bar_utc: str,
        n_bars: int,
        equity: float,
        units: float,
        pending: dict[str, Any] | None,
        trades: list[Any],
        notes: str,
    ) -> int:
        with Session(self.engine) as session:
            session.execute(delete(PaperFill))
            session.execute(delete(PaperSession))
            row = PaperSession(
                symbol=symbol,
                strategy=strategy,
                venue="paper_simulator",
                created_at_utc=datetime.now(UTC),
                last_bar_utc=last_bar_utc,
                n_bars=n_bars,
                equity=equity,
                units=units,
                pending_json=json.dumps(pending or {}, default=str),
                notes=notes,
            )
            session.add(row)
            session.flush()
            for trade in trades:
                session.add(
                    PaperFill(
                        session_id=row.id,
                        strategy=trade.strategy,
                        direction=trade.direction,
                        entry_time=trade.entry_time,
                        exit_time=trade.exit_time,
                        entry_price=trade.entry_price,
                        exit_price=trade.exit_price,
                        units=trade.units,
                        pnl=trade.pnl,
                        costs=trade.costs,
                        exit_reason=trade.exit_reason,
                        venue="paper_simulator",
                    )
                )
            session.commit()
            return int(row.id)

    def latest_session(self) -> PaperSession | None:
        with Session(self.engine) as session:
            return session.scalar(select(PaperSession).order_by(PaperSession.id.desc()))
