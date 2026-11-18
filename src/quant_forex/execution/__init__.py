"""Simulated paper trading. No live broker."""

from quant_forex.execution.broker import LiveTradingNotImplemented, submit_live_order
from quant_forex.execution.pipeline import PaperPipelineResult, run_paper

__all__ = [
    "LiveTradingNotImplemented",
    "PaperPipelineResult",
    "run_paper",
    "submit_live_order",
]
