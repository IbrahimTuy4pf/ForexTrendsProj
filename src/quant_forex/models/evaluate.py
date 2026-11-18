"""Classification metrics and simple importances. Not a trading edge."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = y_true.astype(int)
    pred = (proba >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "precision": _safe(precision_score, y_true, pred),
        "recall": _safe(recall_score, y_true, pred),
        "f1": _safe(f1_score, y_true, pred),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "positive_rate_pred": float(pred.mean()) if len(pred) else None,
        "positive_rate_true": float(y_true.mean()) if len(y_true) else None,
    }
    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    return metrics


def _safe(fn, y_true, pred) -> float | None:
    try:
        return float(fn(y_true, pred, zero_division=0))
    except ValueError:
        return None


def feature_importance(estimator: Any, columns: list[str]) -> list[dict[str, Any]]:
    model = estimator
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps["clf"]
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return []
    rows = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(columns, values, strict=False)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)
