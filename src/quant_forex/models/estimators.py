"""Sklearn classifiers with conservative defaults. Scaler is fit on train only."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_forex.config.schema import ModelConfig


def build_estimator(name: str, config: ModelConfig, seed: int) -> Any:
    if name == "logistic":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model {name!r}")


def predict_proba_up(estimator: Any, x: np.ndarray) -> np.ndarray:
    proba = estimator.predict_proba(x)
    classes = list(estimator.classes_) if hasattr(estimator, "classes_") else list(
        estimator.named_steps["clf"].classes_
    )
    if 1.0 in classes:
        idx = classes.index(1.0)
    elif 1 in classes:
        idx = classes.index(1)
    else:
        idx = 1 if proba.shape[1] > 1 else 0
    return proba[:, idx]
