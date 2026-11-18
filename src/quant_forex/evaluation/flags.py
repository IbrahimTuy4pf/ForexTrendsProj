"""Flags for results that are too clean to trust at face value."""

from __future__ import annotations

from typing import Any

TOO_GOOD_MESSAGE = (
    "Potential overfitting or unrealistic assumptions. Investigate before proceeding."
)


def performance_flags(
    *,
    total_return: float | None,
    max_drawdown: float | None,
    sharpe: float | None,
    win_rate: float | None,
    n_trades: int | None,
    too_good_total_return: float,
    too_good_max_drawdown: float,
    too_good_sharpe: float,
) -> list[str]:
    flags: list[str] = []
    ret = total_return if total_return is not None else 0.0
    dd = max_drawdown if max_drawdown is not None else 1.0
    if ret >= too_good_total_return and dd <= too_good_max_drawdown:
        flags.append(TOO_GOOD_MESSAGE)
    if sharpe is not None and sharpe >= too_good_sharpe and dd <= too_good_max_drawdown:
        flags.append("Sharpe is extreme versus the observed drawdown.")
    if win_rate is not None and n_trades and win_rate >= 0.85 and n_trades >= 10:
        flags.append("Win rate is unusually high; check costs, look-ahead, and sample size.")
    if n_trades is not None and n_trades < 8 and ret > 0:
        flags.append("Very few trades: a positive sample is not statistically meaningful.")
    return flags


def auc_gap_flags(train_auc: float | None, test_auc: float | None, gap_warn: float) -> list[str]:
    if train_auc is None or test_auc is None:
        return []
    if train_auc - test_auc >= gap_warn:
        return [
            f"Train ROC-AUC {train_auc:.3f} exceeds test {test_auc:.3f} by "
            f"{train_auc - test_auc:.3f}. Performance degraded out of sample."
        ]
    return []


def sensitivity_flags(sharpes: list[float | None]) -> list[str]:
    values = [item for item in sharpes if item is not None]
    if len(values) < 4:
        return ["Sensitivity grid is small; do not treat one cell as proof of robustness."]
    best = max(values)
    median = float(sorted(values)[len(values) // 2])
    if best > 0 and best > median + 1.0:
        return [
            "Best Sharpe is an isolated spike versus nearby parameters. "
            "The rule may be fitting noise."
        ]
    if best <= 0 and median <= 0:
        return ["Nearby parameter settings are also weak. That is more honest than a single lucky grid cell."]
    return []


def complexity_notes(models_cfg: Any) -> dict[str, Any]:
    return {
        "n_estimators": models_cfg.n_estimators,
        "max_depth": models_cfg.max_depth,
        "min_samples_leaf": models_cfg.min_samples_leaf,
        "note": (
            "Depth and leaf size are complexity knobs. Raising them to chase in-sample "
            "accuracy is a common overfitting path. This project does not search them on the test fold."
        ),
    }
