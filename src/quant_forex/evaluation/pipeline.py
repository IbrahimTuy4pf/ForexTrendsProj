"""Walk-forward ML evaluation. Each fold refits on past data only."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.evaluation.folds import generate_walk_forward_folds
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.features.engine import build_features
from quant_forex.models.estimators import build_estimator, predict_proba_up
from quant_forex.models.evaluate import fold_metrics
from quant_forex.models.positions import costed_overlay_backtest, proba_to_position
from quant_forex.models.target import TARGET_COL, add_next_return_target, feature_matrix_columns
from quant_forex.utils.paths import ProjectPaths

logger = logging.getLogger("quant_forex.evaluation")


@dataclass
class WalkForwardResult:
    input_path: Path
    report_path: Path
    json_path: Path
    predictions_path: Path
    n_folds: int
    mode: str
    oos_backtest: dict[str, Any] | None


def run_walk_forward(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
) -> WalkForwardResult:
    if input_csv is None:
        found = latest_processed_csv(paths.data_processed)
        if found is None:
            raise FileNotFoundError("No processed OHLCV. Run ingest first.")
        input_path = found
    else:
        candidate = Path(input_csv)
        input_path = candidate if candidate.is_file() else paths.resolve(candidate)

    ohlcv = load_processed_ohlcv(input_path)
    built = build_features(ohlcv, config.features)
    labeled = add_next_return_target(built.frame, config.models)
    feature_cols = feature_matrix_columns(labeled)
    if TARGET_COL in feature_cols:
        raise RuntimeError("Target leaked into feature columns.")
    usable = labeled.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    folds = generate_walk_forward_folds(
        len(usable), config.walk_forward, config.models.horizon
    )
    if len(folds) < 1:
        raise ValueError(
            "Not enough complete rows for walk-forward. Use more history or smaller windows."
        )

    x = usable[feature_cols]
    y = usable[TARGET_COL].astype(int)
    seed = config.project.random_seed
    records: list[dict[str, Any]] = []
    oos_chunks: list[pd.DataFrame] = []

    for fold in folds:
        x_fit = x.iloc[fold.fit_start : fold.fit_end]
        y_fit = y.iloc[fold.fit_start : fold.fit_end]
        x_val = x.iloc[fold.val_start : fold.val_end]
        y_val = y.iloc[fold.val_start : fold.val_end]
        x_test = x.iloc[fold.test_start : fold.test_end]
        y_test = y.iloc[fold.test_start : fold.test_end]
        if x_fit.empty or x_val.empty or x_test.empty:
            continue

        val_scores: dict[str, float] = {}
        fitted: dict[str, Any] = {}
        val_table: dict[str, Any] = {}
        for name in config.models.compare:
            estimator = build_estimator(name, config.models, seed)
            estimator.fit(x_fit.to_numpy(), y_fit.to_numpy())
            fitted[name] = estimator
            val_proba = predict_proba_up(estimator, x_val.to_numpy())
            val_table[name] = fold_metrics(y_val.to_numpy(), val_proba)
            auc = val_table[name]["roc_auc"]
            val_scores[name] = float(auc) if auc is not None else -1.0

        selected = max(val_scores, key=val_scores.get)
        winner = fitted[selected]
        test_proba = predict_proba_up(winner, x_test.to_numpy())
        test_metrics = fold_metrics(y_test.to_numpy(), test_proba)
        positions = proba_to_position(
            test_proba,
            config.models.probability_long,
            config.models.probability_short,
        )
        test_frame = usable.iloc[fold.test_start : fold.test_end].copy()
        chunk = test_frame[["timestamp", "close"]].copy()
        chunk["fold_id"] = fold.fold_id
        chunk["model"] = selected
        chunk["probability_up"] = test_proba
        chunk["position"] = positions
        chunk["y_true"] = y_test.to_numpy()
        oos_chunks.append(chunk)

        fit_times = usable.iloc[fold.fit_start : fold.fit_end]["timestamp"]
        val_times = usable.iloc[fold.val_start : fold.val_end]["timestamp"]
        test_times = test_frame["timestamp"]
        records.append(
            {
                "fold_id": fold.fold_id,
                "selected_model": selected,
                "n_fit": int(len(x_fit)),
                "n_validation": int(len(x_val)),
                "n_test": int(len(x_test)),
                "fit_start": str(fit_times.iloc[0]),
                "fit_end": str(fit_times.iloc[-1]),
                "val_start": str(val_times.iloc[0]),
                "val_end": str(val_times.iloc[-1]),
                "test_start": str(test_times.iloc[0]),
                "test_end": str(test_times.iloc[-1]),
                "val_roc_auc": val_table[selected]["roc_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "validation_metrics": val_table,
                "test_metrics": test_metrics,
            }
        )
        logger.info(
            "fold %s selected=%s val_auc=%s test_auc=%s",
            fold.fold_id,
            selected,
            val_table[selected]["roc_auc"],
            test_metrics["roc_auc"],
        )

    oos = pd.concat(oos_chunks, ignore_index=True)
    if oos["timestamp"].duplicated().any() and config.walk_forward.step_bars < config.walk_forward.test_bars:
        logger.warning(
            "Overlapping test windows: later folds overwrite earlier OOS positions at the same timestamp."
        )
        oos = oos.drop_duplicates(subset=["timestamp"], keep="last")

    oos_backtest = None
    if config.walk_forward.backtest_oos:
        oos_backtest = costed_overlay_backtest(
            ohlcv,
            oos,
            oos["position"].to_numpy(),
            config,
            "walk_forward_oos",
            "Backtest uses concatenated OOS positions only; fit/validation bars stay flat.",
        )

    stem = input_path.stem.replace("_processed", "")
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    pred_path = paths.data_processed / f"{stem}_walkforward_predictions.csv"
    oos.to_csv(pred_path, index=False)
    figure_dir = paths.reports / "figures" / f"{stem}_walkforward"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_fold_auc(records, figure_dir / "oos_auc.png")

    payload = {
        "mode": config.walk_forward.mode,
        "horizon": config.models.horizon,
        "embargo_bars": config.walk_forward.embargo_bars,
        "purge_note": (
            "Train labels that would use prices inside the test window are excluded "
            "(gap = horizon + embargo_bars)."
        ),
        "n_rows": int(len(usable)),
        "n_folds": len(records),
        "folds": records,
        "mean_test_roc_auc": _mean([row["test_roc_auc"] for row in records]),
        "mean_val_roc_auc": _mean([row["val_roc_auc"] for row in records]),
        "oos_backtest": oos_backtest,
        "caveat": (
            "Walk-forward is still not a live result. A stable OOS AUC near 0.5 is the "
            "usual outcome on FX noise."
        ),
    }
    json_path = paths.reports / f"{stem}_walkforward.json"
    report_path = paths.reports / f"{stem}_walkforward.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path.write_text(_markdown(payload), encoding="utf-8")
    return WalkForwardResult(
        input_path=input_path,
        report_path=report_path,
        json_path=json_path,
        predictions_path=pred_path,
        n_folds=len(records),
        mode=config.walk_forward.mode,
        oos_backtest=oos_backtest,
    )


def _mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(np.mean(nums))


def _plot_fold_auc(records: list[dict[str, Any]], path: Path) -> None:
    ids = [row["fold_id"] for row in records]
    test_auc = [row["test_roc_auc"] if row["test_roc_auc"] is not None else np.nan for row in records]
    val_auc = [row["val_roc_auc"] if row["val_roc_auc"] is not None else np.nan for row in records]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ids, val_auc, marker="o", label="validation ROC-AUC")
    ax.plot(ids, test_auc, marker="o", label="OOS ROC-AUC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Walk-forward AUCs (not a profitability claim)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = []
    for row in payload["folds"]:
        lines.append(
            f"| {row['fold_id']} | `{row['selected_model']}` | {row['n_fit']} | "
            f"{_fmt(row['val_roc_auc'])} | {_fmt(row['test_roc_auc'])} | "
            f"{_fmt(row['test_f1'])} | {row['test_start'][:10]} → {row['test_end'][:10]} |"
        )
    bt = payload.get("oos_backtest") or {}
    return f"""# Walk-forward report (Phase 8)

Mode: `{payload['mode']}`  
Folds: {payload['n_folds']}  
Mean validation ROC-AUC: {_fmt(payload.get('mean_val_roc_auc'))}  
Mean OOS ROC-AUC: {_fmt(payload.get('mean_test_roc_auc'))}  

{payload['purge_note']} Model family is chosen on the **validation** slice of each fold. Test bars are never used to fit or select.

| Fold | Selected | Fit rows | Val AUC | OOS AUC | OOS F1 | OOS window |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(lines)}

Concatenated OOS costed backtest (fit/val flat): trades={bt.get('number_of_trades')}, total_return={_fmt(bt.get('total_return'))}, max_dd={_fmt(bt.get('max_drawdown'))}.

{payload['caveat']}
"""
