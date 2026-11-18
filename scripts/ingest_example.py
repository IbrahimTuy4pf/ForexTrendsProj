"""Offline Phase 1 example: ingest the bundled synthetic EURUSD CSV.

This is a convenience wrapper around `python -m quant_forex ingest`.
"""

from __future__ import annotations

from quant_forex.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["ingest", "--config", "configs/default.yaml"]))
