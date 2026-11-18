"""Phase 4 example: run baseline strategies on the latest processed file."""

from __future__ import annotations

from quant_forex.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["baselines", "--config", "configs/default.yaml"]))
