"""Markdown report for a feature build."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_forex.features.engine import FeatureBuildResult


def features_markdown(
    result: FeatureBuildResult,
    association: pd.DataFrame,
    output_csv: Path,
) -> str:
    nan_counts = result.frame[result.feature_columns].isna().sum().sort_values(ascending=False)
    nan_lines = [f"| `{name}` | {int(count)} |" for name, count in nan_counts.items()]
    assoc_lines = []
    for rec in association.head(15).to_dict(orient="records"):
        corr = rec.get("pearson_corr_with_next_return")
        corr_txt = f"{corr:.4f}" if corr is not None and pd.notna(corr) else "n/a"
        assoc_lines.append(f"| `{rec['feature']}` | {rec['n']} | {corr_txt} |")
    notes = "\n".join(f"- {item}" for item in result.warmup_notes)
    return f"""# Feature report (Phase 3)

Output: `{output_csv.as_posix()}`  
Rows before: {result.n_rows_before}  
Rows after: {result.n_rows_after}  
Groups: {", ".join(result.groups)}  
Feature columns: {len(result.feature_columns)}

**No feature is assumed useful.** A large in-sample correlation with the next return is not an edge.

## Notes

{notes}

## Missing values by feature (warmup / gaps)

| Feature | NaN count |
| --- | --- |
{chr(10).join(nan_lines)}

## Univariate |correlation| with next-bar simple return

Target is `close.pct_change().shift(-1)` computed in memory. It is **not** a trading signal.
Pearson correlation is a crude screen only. Do not select features on the full sample and then test on the same rows.

| Feature | n | Pearson corr |
| --- | --- | --- |
{chr(10).join(assoc_lines) if assoc_lines else "| (none) | | |"}
"""
