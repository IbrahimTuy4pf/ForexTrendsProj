"""Phase 3 example: build features from the latest processed dataset."""

from __future__ import annotations

from quant_forex.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["features", "--config", "configs/default.yaml"]))
