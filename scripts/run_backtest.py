"""Phase 5 example: costed backtest of enabled baselines."""

from __future__ import annotations

from quant_forex.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["backtest", "--config", "configs/default.yaml"]))
