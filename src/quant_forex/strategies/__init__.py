"""Baseline rule-based strategies (Phase 4). Not a backtest engine."""

from quant_forex.strategies.pipeline import BaselinePipelineResult, run_baselines
from quant_forex.strategies.signal import TradingSignal

__all__ = ["BaselinePipelineResult", "TradingSignal", "run_baselines"]
