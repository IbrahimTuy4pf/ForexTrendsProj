"""Write the bundled synthetic EURUSD CSV used by configs/default.yaml."""

from __future__ import annotations

from pathlib import Path

from quant_forex.data.sample import generate_synthetic_ohlcv
from quant_forex.utils.paths import ProjectPaths


def main() -> None:
    paths = ProjectPaths.find()
    out = paths.data_external / "sample_eurusd_daily.csv"
    frame = generate_synthetic_ohlcv(n_rows=260, start="2024-01-02", seed=42)
    export = frame.copy()
    export["timestamp"] = export["timestamp"].dt.strftime("%Y-%m-%d")
    out.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out, index=False)
    header = (
        "# SYNTHETIC DATA — not real EURUSD. Generated for offline pipeline tests.\n"
        "# Do not use this file to claim market findings.\n"
    )
    body = out.read_text(encoding="utf-8")
    if not body.startswith("#"):
        out.write_text(header + body, encoding="utf-8")
    print(f"Wrote {out} ({len(export)} rows)")


if __name__ == "__main__":
    main()
