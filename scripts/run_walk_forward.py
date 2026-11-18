"""Phase 8 example: walk-forward classifiers on processed OHLCV."""

from __future__ import annotations

from quant_forex.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["walk-forward", "--config", "configs/default.yaml"]))
