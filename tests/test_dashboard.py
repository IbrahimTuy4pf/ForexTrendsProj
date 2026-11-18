from __future__ import annotations

import json

import pandas as pd

from quant_forex.dashboard.catalog import (
    backtest_metrics_table,
    catalog,
    latest_report,
    load_equity,
    load_json,
    load_ohlcv_preview,
)
from quant_forex.dashboard.launch import app_path
from quant_forex.data.sample import generate_synthetic_ohlcv


def test_app_path_exists() -> None:
    assert app_path().is_file()


def test_catalog_discovers_reports_and_processed(project_paths) -> None:
    processed = project_paths.data_processed / "EURUSD_1d_demo_processed.csv"
    frame = generate_synthetic_ohlcv(n_rows=15, seed=61)
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    export.to_csv(processed, index=False)
    equity = project_paths.data_processed / "EURUSD_1d_demo_backtest_equity.csv"
    pd.DataFrame(
        {
            "strategy": ["ma_crossover", "ma_crossover"],
            "timestamp": ["2024-01-02", "2024-01-03"],
            "equity": [10000.0, 10010.0],
            "drawdown": [0.0, 0.0],
        }
    ).to_csv(equity, index=False)
    (project_paths.reports / "EURUSD_1d_demo_backtest.json").write_text(
        json.dumps(
            {
                "metrics": [
                    {
                        "strategy": "ma_crossover",
                        "number_of_trades": 2,
                        "total_return": 0.01,
                        "max_drawdown": 0.02,
                        "sharpe_ratio": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    snap = catalog(project_paths)
    assert snap["live_trading_allowed"] is False
    assert snap["processed_csv"] == processed
    assert snap["equity_csv"] == equity
    assert latest_report(project_paths.reports, "backtest") is not None
    table = backtest_metrics_table(load_json(snap["reports"]["backtest"]))
    assert list(table["strategy"]) == ["ma_crossover"]
    preview = load_ohlcv_preview(processed)
    assert "close" in preview.columns
    loaded = load_equity(equity)
    assert loaded["equity"].iloc[-1] == 10010.0
