"""Build and persist a feature dataset from processed OHLCV."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from quant_forex.config.schema import AppConfig
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.features.engine import FeatureBuildResult, build_features
from quant_forex.features.importance import next_bar_associations
from quant_forex.features.report import features_markdown
from quant_forex.utils.paths import ProjectPaths


@dataclass
class FeaturePipelineResult:
    input_path: Path
    output_csv: Path
    report_path: Path
    json_path: Path
    n_rows: int
    n_features: int
    build: FeatureBuildResult


def run_feature_pipeline(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
) -> FeaturePipelineResult:
    if input_csv is None:
        found = latest_processed_csv(paths.data_processed)
        if found is None:
            raise FileNotFoundError(
                "No *_processed.csv in data/processed. Run ingest first."
            )
        input_path = found
    else:
        candidate = Path(input_csv)
        input_path = candidate if candidate.is_file() else paths.resolve(candidate)

    ohlcv = load_processed_ohlcv(input_path)
    build = build_features(ohlcv, config.features)
    stem = input_path.stem.replace("_processed", "")
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    output_csv = paths.data_processed / f"{stem}_features.csv"
    build.frame.to_csv(output_csv, index=False)

    association = next_bar_associations(build.frame, build.feature_columns)
    report_path = paths.reports / f"{stem}_features.md"
    json_path = paths.reports / f"{stem}_features.json"
    try:
        csv_display = output_csv.relative_to(paths.root)
    except ValueError:
        csv_display = output_csv
    report_path.write_text(
        features_markdown(build, association, csv_display),
        encoding="utf-8",
    )
    payload = {
        "input_path": str(input_path),
        "output_csv": str(output_csv),
        "groups": build.groups,
        "feature_columns": build.feature_columns,
        "n_rows_before": build.n_rows_before,
        "n_rows_after": build.n_rows_after,
        "warmup_notes": build.warmup_notes,
        "associations": association.to_dict(orient="records"),
        "config": config.features.model_dump(mode="json"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return FeaturePipelineResult(
        input_path=input_path,
        output_csv=output_csv,
        report_path=report_path,
        json_path=json_path,
        n_rows=build.n_rows_after,
        n_features=len(build.feature_columns),
        build=build,
    )
