"""Train simple classifiers with chronological splits. Selection uses validation only."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quant_forex.config.schema import AppConfig
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.features.engine import build_features
from quant_forex.models.estimators import build_estimator, predict_proba_up
from quant_forex.models.evaluate import feature_importance, fold_metrics
from quant_forex.models.positions import costed_overlay_backtest, proba_to_position
from quant_forex.models.splits import chronological_masks
from quant_forex.models.target import TARGET_COL, add_next_return_target, feature_matrix_columns
from quant_forex.strategies.signal import TradingSignal
from quant_forex.utils.paths import ProjectPaths

logger = logging.getLogger("quant_forex.models")


@dataclass
class MLPipelineResult:
    input_path: Path
    report_path: Path
    json_path: Path
    predictions_path: Path
    artifact_path: Path
    selected_model: str
    split_method: str
    metrics: dict[str, Any]
    signals: list[TradingSignal] = field(default_factory=list)


def run_ml_pipeline(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
) -> MLPipelineResult:
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
    if len(usable) < 40:
        raise ValueError("Not enough complete rows after warmup/target drop for ML.")

    train_m, val_m, test_m, split_method = chronological_masks(
        usable["timestamp"], config.splits
    )
    x = usable[feature_cols]
    y = usable[TARGET_COL].astype(int)
    x_train, y_train = x.loc[train_m], y.loc[train_m]
    x_val, y_val = x.loc[val_m], y.loc[val_m]
    x_test, y_test = x.loc[test_m], y.loc[test_m]
    if len(x_train) < 20 or len(x_val) < 5 or len(x_test) < 5:
        raise ValueError("A chronological fold is too small. Use more history.")

    seed = config.project.random_seed
    fold_tables: dict[str, Any] = {}
    val_scores: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    importances: dict[str, list[dict[str, Any]]] = {}

    for name in config.models.compare:
        estimator = build_estimator(name, config.models, seed)
        estimator.fit(x_train.to_numpy(), y_train.to_numpy())
        fitted[name] = estimator
        val_proba = predict_proba_up(estimator, x_val.to_numpy())
        test_proba = predict_proba_up(estimator, x_test.to_numpy())
        train_proba = predict_proba_up(estimator, x_train.to_numpy())
        fold_tables[name] = {
            "train": fold_metrics(y_train.to_numpy(), train_proba),
            "validation": fold_metrics(y_val.to_numpy(), val_proba),
            "test": fold_metrics(y_test.to_numpy(), test_proba),
        }
        auc = fold_tables[name]["validation"]["roc_auc"]
        val_scores[name] = float(auc) if auc is not None else -1.0
        importances[name] = feature_importance(estimator, feature_cols)
        logger.info("%s validation roc_auc=%s", name, auc)

    selected = max(val_scores, key=val_scores.get)
    winner = fitted[selected]
    test_proba = predict_proba_up(winner, x_test.to_numpy())
    positions_test = proba_to_position(
        test_proba, config.models.probability_long, config.models.probability_short
    )
    signals = _signals_from_test(
        usable.loc[test_m].reset_index(drop=True),
        test_proba,
        positions_test,
        importances[selected][:5],
        selected,
        config.instrument.symbol,
    )
    test_backtest = None
    if config.models.backtest_test_fold:
        test_backtest = costed_overlay_backtest(
            ohlcv,
            usable.loc[test_m],
            positions_test,
            config,
            f"ml_{selected}",
            "Backtest uses ML positions on the TEST fold only; train/validation bars stay flat.",
        )

    stem = input_path.stem.replace("_processed", "")
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    pred_path = paths.data_processed / f"{stem}_ml_predictions.csv"
    pred = usable.loc[test_m, ["timestamp", "close"]].copy()
    pred["probability_up"] = test_proba
    pred["position"] = positions_test
    pred["model"] = selected
    pred.to_csv(pred_path, index=False)

    artifact_path = paths.data_processed / f"{stem}_ml_{selected}.joblib"
    joblib.dump(
        {
            "estimator": winner,
            "feature_columns": feature_cols,
            "model_name": selected,
            "horizon": config.models.horizon,
            "threshold": config.models.threshold,
            "probability_long": config.models.probability_long,
            "probability_short": config.models.probability_short,
            "split_method": split_method,
        },
        artifact_path,
    )

    report_path = paths.reports / f"{stem}_ml.md"
    json_path = paths.reports / f"{stem}_ml.json"
    payload = {
        "target": {
            "name": TARGET_COL,
            "description": (
                f"1 if close[t+{config.models.horizon}]/close[t]-1 > {config.models.threshold}"
            ),
            "horizon": config.models.horizon,
            "threshold": config.models.threshold,
        },
        "split_method": split_method,
        "n_train": int(train_m.sum()),
        "n_validation": int(val_m.sum()),
        "n_test": int(test_m.sum()),
        "selected_by": "validation_roc_auc",
        "selected_model": selected,
        "feature_columns": feature_cols,
        "metrics": fold_tables,
        "feature_importance": importances,
        "artifact": str(artifact_path),
        "test_backtest": test_backtest,
        "caveat": "A high training score with a weak test score is a warning, not a strategy.",
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path.write_text(_markdown(payload, signals), encoding="utf-8")
    return MLPipelineResult(
        input_path=input_path,
        report_path=report_path,
        json_path=json_path,
        predictions_path=pred_path,
        artifact_path=artifact_path,
        selected_model=selected,
        split_method=split_method,
        metrics=fold_tables,
        signals=signals,
    )


def _signals_from_test(
    test_frame: pd.DataFrame,
    proba: np.ndarray,
    position: np.ndarray,
    top_features: list[dict[str, Any]],
    model_name: str,
    symbol: str,
) -> list[TradingSignal]:
    signals: list[TradingSignal] = []
    prev = 0.0
    for i, row in test_frame.reset_index(drop=True).iterrows():
        pos = float(position[i])
        if pos == 0 or pos == prev:
            prev = pos
            continue
        direction = "LONG" if pos > 0 else "SHORT"
        ts = row["timestamp"]
        ts_text = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        p = float(proba[i])
        signals.append(
            TradingSignal(
                timestamp=ts_text,
                instrument=symbol,
                direction=direction,
                probability=p if pos > 0 else 1.0 - p,
                entry=float(row["close"]),
                stop_loss=None,
                take_profit=None,
                position_size=None,
                risk_percentage=None,
                model_version=model_name,
                strategy_name=f"ml_{model_name}",
                strategy_version="1.0.0",
                reason=(
                    f"P(next return > threshold)={p:.3f}. "
                    f"Top features (train importances): "
                    + ", ".join(item["feature"] for item in top_features)
                ),
                features={item["feature"]: item["importance"] for item in top_features},
            )
        )
        prev = pos
    return signals


def _markdown(payload: dict[str, Any], signals: list[TradingSignal]) -> str:
    lines = []
    for name, folds in payload["metrics"].items():
        val = folds["validation"]
        test = folds["test"]
        mark = " ← selected" if name == payload["selected_model"] else ""
        lines.append(
            f"| `{name}`{mark} | {_fmt(val.get('roc_auc'))} | {_fmt(test.get('roc_auc'))} | "
            f"{_fmt(test.get('f1'))} | {_fmt(test.get('precision'))} | {_fmt(test.get('recall'))} |"
        )
    example = ""
    if signals:
        sig = signals[0]
        example = (
            f"\n## Example test-fold signal\n\n"
            f"Prediction: **{sig.direction}**  \n"
            f"Probability: `{sig.probability}`  \n"
            f"Entry: `{sig.entry}`  \n"
            f"Reason: {sig.reason}\n"
        )
    top = payload["feature_importance"].get(payload["selected_model"], [])[:8]
    imp = "\n".join(f"| `{row['feature']}` | {row['importance']:.4f} |" for row in top)
    bt = payload.get("test_backtest") or {}
    return f"""# ML report (Phase 7)

Target: `{payload['target']['description']}`  
Split: `{payload['split_method']}` (train {payload['n_train']} / val {payload['n_validation']} / test {payload['n_test']})  
Selected model: `{payload['selected_model']}` by **validation ROC-AUC only**.

This is a research classifier, not a profitable system. Do not shuffle time series. The scaler (logistic) is fit on train only. Hyperparameters were not searched on the test fold.

| Model | Val ROC-AUC | Test ROC-AUC | Test F1 | Test precision | Test recall |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(lines)}

Test-fold costed backtest (train/val flat): trades={bt.get('number_of_trades')}, total_return={_fmt(bt.get('total_return'))}, max_dd={_fmt(bt.get('max_drawdown'))}.

## Selected-model importances (train)

| Feature | Importance |
| --- | --- |
{imp or '| (none) | |'}
{example}
A large train/test AUC gap is a warning of overfitting. Prefer `python -m quant_forex walk-forward` over a single split.
"""


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
