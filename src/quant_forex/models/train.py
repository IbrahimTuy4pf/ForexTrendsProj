"""Entry point used by the CLI. Fitting happens in pipeline.py."""

from quant_forex.models.pipeline import MLPipelineResult, run_ml_pipeline

__all__ = ["MLPipelineResult", "run_ml_pipeline"]
