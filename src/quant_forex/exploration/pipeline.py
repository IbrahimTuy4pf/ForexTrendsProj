"""Run Phase 2 exploration on a processed OHLCV file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from quant_forex.config.schema import AppConfig
from quant_forex.exploration.gaps import gap_table
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.exploration.plots import save_exploration_charts
from quant_forex.exploration.prepare import add_descriptive_columns
from quant_forex.exploration.report import exploration_markdown
from quant_forex.exploration.stats import build_exploration_summary
from quant_forex.exploration.summary import ExplorationSummary
from quant_forex.utils.paths import ProjectPaths


@dataclass
class ExplorationResult:
    input_path: Path
    report_path: Path
    json_path: Path
    figure_dir: Path
    figure_paths: list[Path]
    summary: ExplorationSummary
    n_rows: int


def run_exploration(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
    source_note: str | None = None,
) -> ExplorationResult:
    if input_csv is None:
        found = latest_processed_csv(paths.data_processed)
        if found is None:
            raise FileNotFoundError(
                "No *_processed.csv in data/processed. Run `python -m quant_forex ingest` first."
            )
        input_path = found
    else:
        candidate = Path(input_csv)
        input_path = candidate if candidate.is_file() else paths.resolve(candidate)

    raw = load_processed_ohlcv(input_path)
    frame = add_descriptive_columns(raw)
    symbol = str(frame["symbol"].iloc[0]) if "symbol" in frame.columns else config.instrument.symbol
    timeframe = (
        str(frame["timeframe"].iloc[0]) if "timeframe" in frame.columns else config.timeframe
    )

    note = source_note
    csv_hint = str(config.data.csv_path or "")
    if note is None and (
        "sample_eurusd" in input_path.name.lower() or "sample_eurusd" in csv_hint.lower()
    ):
        note = "Input is configured as the bundled synthetic sample, not real EURUSD."
    stem = input_path.stem.replace("_processed", "")
    figure_dir = paths.reports / "figures" / stem
    figure_paths = save_exploration_charts(
        frame,
        figure_dir,
        symbol=symbol,
        rolling_window=config.exploration.rolling_window,
        save_plotly_html=config.exploration.save_plotly_html,
    ) if config.exploration.save_matplotlib else []

    summary = build_exploration_summary(
        frame,
        symbol=symbol,
        timeframe=timeframe,
        expected_frequency=config.data.expected_frequency,
        autocorr_lags=config.exploration.autocorrelation_lags,
        source_note=note,
    )
    gaps = gap_table(frame["timestamp"], config.data.expected_frequency)
    report_path = paths.reports / f"{stem}_exploration.md"
    json_path = paths.reports / f"{stem}_exploration.json"
    markdown = exploration_markdown(summary, figure_paths, relative_to=paths.reports)
    if not gaps.empty:
        markdown += "\n## Detected gaps vs expected calendar\n\n"
        markdown += "| gap_start | gap_end | n_missing_bars |\n| --- | --- | --- |\n"
        for rec in gaps.to_dict(orient="records"):
            markdown += (
                f"| {rec['gap_start']} | {rec['gap_end']} | {rec['n_missing_bars']} |\n"
            )
    paths.reports.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    payload = summary.model_dump(mode="json")
    payload["input_path"] = str(input_path)
    payload["gaps"] = gaps.to_dict(orient="records")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ExplorationResult(
        input_path=input_path,
        report_path=report_path,
        json_path=json_path,
        figure_dir=figure_dir,
        figure_paths=figure_paths,
        summary=summary,
        n_rows=len(frame),
    )
