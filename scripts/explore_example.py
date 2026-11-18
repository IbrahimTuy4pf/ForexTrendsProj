"""Phase 2 example: explore the latest processed dataset."""

from __future__ import annotations

from quant_forex.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["explore", "--config", "configs/default.yaml"]))
