"""Read local research artifacts for the dashboard. No broker, no live orders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.utils.paths import ProjectPaths

REPORT_KINDS = (
    "data_quality",
    "exploration",
    "features",
    "baselines",
    "backtest",
    "ml",
    "walkforward",
    "diagnostics",
    "paper",
)


def latest_report(reports_dir: Path, kind: str) -> Path | None:
    files = sorted(reports_dir.glob(f"*_{kind}.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sibling_markdown(json_path: Path | None) -> str | None:
    if json_path is None:
        return None
    md = json_path.with_suffix(".md")
    if not md.is_file():
        return None
    return md.read_text(encoding="utf-8")


def catalog(paths: ProjectPaths) -> dict[str, Any]:
    reports = {kind: latest_report(paths.reports, kind) for kind in REPORT_KINDS}
    processed = latest_processed_csv(paths.data_processed)
    equity = None
    if processed is not None:
        stem = processed.stem.replace("_processed", "")
        candidate = paths.data_processed / f"{stem}_backtest_equity.csv"
        if candidate.is_file():
            equity = candidate
    return {
        "processed_csv": processed,
        "equity_csv": equity,
        "reports": reports,
        "live_trading_allowed": False,
        "disclaimer": (
            "Research dashboard only. Charts are not live prices, paper equity is simulated, "
            "and this UI cannot place broker orders."
        ),
    }


def load_ohlcv_preview(path: Path | None, n_tail: int = 250) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    frame = load_processed_ohlcv(path)
    return frame.tail(n_tail).reset_index(drop=True)


def load_equity(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def backtest_metrics_table(payload: dict[str, Any] | None) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    rows = payload.get("metrics") or []
    if not rows:
        return pd.DataFrame()
    keep = [
        "strategy",
        "number_of_trades",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "profit_factor",
        "final_equity",
    ]
    frame = pd.DataFrame(rows)
    cols = [col for col in keep if col in frame.columns]
    return frame[cols]
