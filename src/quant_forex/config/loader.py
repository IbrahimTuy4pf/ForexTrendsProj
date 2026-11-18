"""Load and validate YAML configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from quant_forex.config.schema import AppConfig, ExecutionEnvSettings


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config, then apply the environment live-trading guard."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = AppConfig.model_validate(raw)
    ExecutionEnvSettings()
    return config
