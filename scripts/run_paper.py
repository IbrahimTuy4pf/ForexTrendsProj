"""Phase 10 example: simulated paper book (requires --enable-paper)."""

from __future__ import annotations

from quant_forex.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(["paper", "--config", "configs/default.yaml", "--enable-paper", "--strategy", "ma_crossover"])
    )
