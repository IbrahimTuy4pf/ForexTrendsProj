"""Simulated paper session: same fills as the backtest, no broker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_forex.backtesting.engine import run_backtest
from quant_forex.backtesting.metrics import compute_metrics
from quant_forex.config.schema import AppConfig
from quant_forex.execution.broker import submit_live_order
from quant_forex.execution.ledger import PaperLedger
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.strategies.common import stops_from_atr, true_range_atr
from quant_forex.strategies.pipeline import enabled_strategies
from quant_forex.utils.paths import ProjectPaths

VENUE = "paper_simulator"
NO_BROKER_NOTE = (
    "SIMULATED paper book. No broker, no MT4/MT5, no REST order. "
    "LIVE_TRADING remains disabled. A paper PnL is not a live result."
)


@dataclass
class PaperPipelineResult:
    input_path: Path
    report_path: Path
    json_path: Path
    ledger_path: Path
    strategy: str
    equity: float
    units: float
    pending: dict[str, Any] | None
    n_trades: int


def run_paper(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
    strategy_name: str | None = None,
) -> PaperPipelineResult:
    if config.execution.live_trading:
        submit_live_order()
    if not config.execution.paper_trading:
        raise PermissionError(
            "Paper trading is opt-in. Pass --enable-paper or set execution.paper_trading: true. "
            "That still does not contact a broker."
        )

    if input_csv is None:
        found = latest_processed_csv(paths.data_processed)
        if found is None:
            raise FileNotFoundError("No processed OHLCV. Run ingest first.")
        input_path = found
    else:
        candidate = Path(input_csv)
        input_path = candidate if candidate.is_file() else paths.resolve(candidate)

    ohlcv = load_processed_ohlcv(input_path)
    strategies = enabled_strategies(config.strategies)
    chosen = strategy_name or config.paper.strategy
    if chosen:
        strategies = [item for item in strategies if item.name == chosen]
    if not strategies:
        raise ValueError("No enabled strategy to paper-trade. Check --strategy and YAML flags.")
    strategy = strategies[0]

    atr = true_range_atr(ohlcv, config.strategies.atr_window)
    realized_vol = (
        ohlcv["close"]
        .pct_change()
        .rolling(config.risk.vol_lookback, min_periods=config.risk.vol_lookback)
        .std()
    )
    pos_frame = strategy.positions(ohlcv)
    stop, take = stops_from_atr(
        pos_frame["position"],
        ohlcv["close"],
        atr,
        config.strategies.stop_atr_multiple,
        config.strategies.take_profit_atr_multiple,
    )
    result = run_backtest(
        ohlcv,
        pos_frame["position"],
        stop=stop,
        take=take,
        config=config,
        strategy_name=strategy.name,
        realized_vol=realized_vol,
        flatten_at_end=False,
    )
    metrics = compute_metrics(result)
    last = result.equity_curve.iloc[-1]
    last_bar = pd.to_datetime(ohlcv["timestamp"].iloc[-1], utc=True)
    desired = float(pos_frame["position"].iloc[-1])
    held_units = float(last["units"])
    pending = None
    desired_dir = 0 if desired == 0 else (1 if desired > 0 else -1)
    held_dir = 0 if held_units == 0 else (1 if held_units > 0 else -1)
    if desired_dir != held_dir:
        pending = {
            "status": "pending_next_open",
            "venue": VENUE,
            "asof": last_bar.isoformat(),
            "desired_position": desired_dir,
            "held_units": held_units,
            "reason": "Signal is decided at this close; the paper book fills at the next bar open.",
        }

    ledger_path = paths.resolve(config.paper.ledger_path)
    ledger = PaperLedger(ledger_path)
    symbol = (
        str(ohlcv["symbol"].iloc[0])
        if "symbol" in ohlcv.columns
        else config.instrument.symbol
    )
    ledger.replace_session(
        symbol=symbol,
        strategy=strategy.name,
        last_bar_utc=last_bar.isoformat(),
        n_bars=len(ohlcv),
        equity=float(last["equity"]),
        units=held_units,
        pending=pending,
        trades=result.trades,
        notes=NO_BROKER_NOTE,
    )

    stem = input_path.stem.replace("_processed", "")
    paths.reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "venue": VENUE,
        "broker": None,
        "live_trading": False,
        "note": NO_BROKER_NOTE,
        "strategy": strategy.name,
        "symbol": symbol,
        "last_bar": last_bar.isoformat(),
        "equity": float(last["equity"]),
        "open_units": held_units,
        "pending": pending,
        "metrics": {key: _jsonable(val) for key, val in metrics.items()},
        "n_closed_trades": len(result.trades),
        "halted": result.halted,
        "halt_reason": result.halt_reason,
        "ledger": str(ledger_path),
    }
    json_path = paths.reports / f"{stem}_paper.json"
    report_path = paths.reports / f"{stem}_paper.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path.write_text(_markdown(payload), encoding="utf-8")
    return PaperPipelineResult(
        input_path=input_path,
        report_path=report_path,
        json_path=json_path,
        ledger_path=ledger_path,
        strategy=strategy.name,
        equity=float(last["equity"]),
        units=held_units,
        pending=pending,
        n_trades=len(result.trades),
    )


def _jsonable(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    pending = payload.get("pending")
    pending_txt = json.dumps(pending, indent=2) if pending else "none (held position matches last close signal, or flat)"
    return f"""# Paper session (Phase 10)

**{payload['note']}**

Venue: `{payload['venue']}`  
Broker: `{payload['broker']}`  
Strategy: `{payload['strategy']}`  
Symbol: `{payload['symbol']}`  
As of last bar: `{payload['last_bar']}`

| Book | Value |
| --- | --- |
| Simulated equity | {_fmt(payload.get('equity'))} |
| Open units | {_fmt(payload.get('open_units'))} |
| Closed paper trades | {payload.get('n_closed_trades')} |
| Halted | {payload.get('halted')} |

Pending next-open intent:

```json
{pending_txt}
```

Fills use the same next-open + spread/slippage model as the research backtest. The last bar is **not** force-flattened (unlike a completed backtest), so an open paper position can remain.

Ledger: `{payload.get('ledger')}`
"""
