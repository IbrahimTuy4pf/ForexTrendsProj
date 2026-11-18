"""Save matplotlib PNGs and optional Plotly HTML. Charts are descriptive, not P&L."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from quant_forex.exploration.stats import cumulative_log_return, rolling_return_volatility


def save_exploration_charts(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    symbol: str,
    rolling_window: int,
    save_plotly_html: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.append(_price_chart(frame, output_dir, symbol))
    written.append(_returns_hist(frame, output_dir, symbol))
    written.append(_rolling_vol(frame, output_dir, symbol, rolling_window))
    written.append(_weekday_box(frame, output_dir, symbol))
    written.append(_cumulative_log(frame, output_dir, symbol))
    if save_plotly_html:
        written.append(_candlestick_html(frame, output_dir, symbol))
    return [path for path in written if path is not None]


def _price_chart(frame: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    path = output_dir / "close_price.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["timestamp"], frame["close"], color="#1f4e79", linewidth=1.2)
    ax.set_title(f"{symbol} close (descriptive — not a backtest)")
    ax.set_xlabel("timestamp (UTC)")
    ax.set_ylabel("close")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _returns_hist(frame: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    path = output_dir / "simple_returns_hist.png"
    returns = frame["simple_return"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(returns, bins=40, color="#4c78a8", edgecolor="white")
    ax.set_title(f"{symbol} close-to-close simple returns")
    ax.set_xlabel("simple return")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _rolling_vol(frame: pd.DataFrame, output_dir: Path, symbol: str, window: int) -> Path:
    path = output_dir / "rolling_volatility.png"
    vol = rolling_return_volatility(frame["simple_return"], window)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["timestamp"], vol, color="#b35c00", linewidth=1.2)
    ax.set_title(f"{symbol} rolling {window}-bar return std (no future bars)")
    ax.set_xlabel("timestamp (UTC)")
    ax.set_ylabel("std")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _weekday_box(frame: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    path = output_dir / "returns_by_weekday.png"
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    data = [
        frame.loc[frame["day_of_week"] == day, "simple_return"].dropna().to_numpy()
        for day in order
        if (frame["day_of_week"] == day).any()
    ]
    labels = [day[:3] for day in order if (frame["day_of_week"] == day).any()]
    fig, ax = plt.subplots(figsize=(8, 4))
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_title(f"{symbol} return distribution by weekday")
    ax.set_ylabel("simple return")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _cumulative_log(frame: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    path = output_dir / "cumulative_log_return.png"
    curve = cumulative_log_return(frame)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame["timestamp"], curve, color="#2a9d8f", linewidth=1.2)
    ax.set_title(f"{symbol} cumulative log return of spot (NOT a trading P&L)")
    ax.set_xlabel("timestamp (UTC)")
    ax.set_ylabel("sum of log returns")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _candlestick_html(frame: pd.DataFrame, output_dir: Path, symbol: str) -> Path | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return None
    path = output_dir / "candlestick.html"
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=frame["timestamp"],
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name=symbol,
            )
        ]
    )
    fig.update_layout(
        title=f"{symbol} OHLC candlesticks (research view)",
        xaxis_title="timestamp (UTC)",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    fig.write_html(path, include_plotlyjs="cdn")
    return path
