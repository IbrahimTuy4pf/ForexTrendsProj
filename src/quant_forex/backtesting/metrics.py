"""Performance numbers from a finished backtest. Not a claim of live expectancy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_forex.backtesting.engine import BacktestResult, Trade


def trade_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([item.__dict__ for item in trades])


def compute_metrics(result: BacktestResult, periods_per_year: int = 252) -> dict[str, float | int | None]:
    equity = result.equity_curve["equity"].astype(float)
    rets = equity.pct_change().dropna()
    trades = result.trades
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total_return = float(equity.iloc[-1] / result.initial_capital - 1.0) if len(equity) else 0.0
    n = max(len(rets), 1)
    years = n / periods_per_year
    ann = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 and total_return > -1 else None
    vol = float(rets.std(ddof=1) * math.sqrt(periods_per_year)) if len(rets) > 1 else None
    downside = rets[rets < 0]
    down_dev = (
        float(downside.std(ddof=1) * math.sqrt(periods_per_year)) if len(downside) > 1 else None
    )
    sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year)) if len(rets) > 1 and rets.std() else None
    sortino = (
        float(rets.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))
        if len(downside) > 1 and downside.std()
        else None
    )
    max_dd = float(result.equity_curve["drawdown"].max()) if not result.equity_curve.empty else 0.0
    calmar = (ann / max_dd) if ann is not None and max_dd > 0 else None
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None
    n_trades = len(trades)
    win_rate = float(len(wins) / n_trades) if n_trades else None
    return {
        "total_return": total_return,
        "annualized_return": ann,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "loss_rate": (1.0 - win_rate) if win_rate is not None else None,
        "profit_factor": profit_factor,
        "average_winning_trade": float(wins.mean()) if len(wins) else None,
        "average_losing_trade": float(losses.mean()) if len(losses) else None,
        "expectancy": float(pnls.mean()) if n_trades else None,
        "number_of_trades": n_trades,
        "average_trade_duration_bars": float(np.mean([t.bars_held for t in trades])) if n_trades else None,
        "largest_winning_trade": float(wins.max()) if len(wins) else None,
        "largest_losing_trade": float(losses.min()) if len(losses) else None,
        "volatility": vol,
        "downside_deviation": down_dev,
        "final_equity": result.final_equity,
    }
