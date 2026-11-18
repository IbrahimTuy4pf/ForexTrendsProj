"""Phase 9 diagnostics: statistics, sensitivity, and overfitting flags."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_forex.backtesting.engine import run_backtest
from quant_forex.config.schema import AppConfig
from quant_forex.evaluation.flags import (
    auc_gap_flags,
    complexity_notes,
    performance_flags,
    sensitivity_flags,
)
from quant_forex.evaluation.sensitivity import ma_crossover_grid, momentum_grid, run_costed_strategy
from quant_forex.evaluation.stats import bar_returns, bootstrap_summary, calendar_tables, sharpe_ratio
from quant_forex.exploration.io import latest_processed_csv, load_processed_ohlcv
from quant_forex.strategies.common import stops_from_atr, true_range_atr
from quant_forex.strategies.pipeline import enabled_strategies
from quant_forex.utils.paths import ProjectPaths

logger = logging.getLogger("quant_forex.evaluation.diagnose")


@dataclass
class DiagnoseResult:
    input_path: Path
    report_path: Path
    json_path: Path
    n_strategies: int
    flags: list[str]


def run_diagnostics(
    config: AppConfig,
    paths: ProjectPaths,
    input_csv: str | Path | None = None,
) -> DiagnoseResult:
    if input_csv is None:
        found = latest_processed_csv(paths.data_processed)
        if found is None:
            raise FileNotFoundError("No processed OHLCV. Run ingest first.")
        input_path = found
    else:
        candidate = Path(input_csv)
        input_path = candidate if candidate.is_file() else paths.resolve(candidate)

    ohlcv = load_processed_ohlcv(input_path)
    rng = np.random.default_rng(config.project.random_seed)
    stem = input_path.stem.replace("_processed", "")
    figure_dir = paths.reports / "figures" / f"{stem}_diagnostics"
    figure_dir.mkdir(parents=True, exist_ok=True)

    strategy_rows: list[dict[str, Any]] = []
    all_flags: list[str] = []
    ppy = config.risk.periods_per_year
    diag = config.diagnostics

    for strategy in enabled_strategies(config.strategies):
        result, metrics, position = run_costed_strategy(ohlcv, config, strategy)
        rets = bar_returns(result.equity_curve["equity"])
        boot = bootstrap_summary(
            rets,
            n_bootstrap=diag.n_bootstrap,
            block_size=diag.block_size,
            confidence_level=diag.confidence_level,
            seed=config.project.random_seed,
            periods_per_year=ppy,
        )
        perm = _permutation_sharpe(
            ohlcv,
            config,
            strategy,
            position.to_numpy(dtype=float),
            observed=boot.get("sharpe") if isinstance(boot.get("sharpe"), float) else None,
            n_perm=diag.n_permutations,
            rng=rng,
            periods_per_year=ppy,
        )
        flags = performance_flags(
            total_return=_as_float(metrics.get("total_return")),
            max_drawdown=_as_float(metrics.get("max_drawdown")),
            sharpe=_as_float(metrics.get("sharpe_ratio")),
            win_rate=_as_float(metrics.get("win_rate")),
            n_trades=int(metrics["number_of_trades"]) if metrics.get("number_of_trades") else 0,
            too_good_total_return=diag.too_good_total_return,
            too_good_max_drawdown=diag.too_good_max_drawdown,
            too_good_sharpe=diag.too_good_sharpe,
        )
        all_flags.extend(f"{strategy.name}: {item}" for item in flags)
        dist = _return_drawdown_summary(result.equity_curve)
        calendar = calendar_tables(result.equity_curve)
        _plot_return_hist(rets, figure_dir / f"{strategy.name}_returns.png", strategy.name)
        strategy_rows.append(
            {
                "strategy": strategy.name,
                "metrics": {key: _jsonable(val) for key, val in metrics.items()},
                "bootstrap": _jsonable(boot),
                "permutation": perm,
                "distributions": dist,
                "calendar": calendar,
                "flags": flags,
            }
        )
        logger.info("diagnosed %s flags=%s", strategy.name, flags)

    sensitivity: dict[str, Any] = {}
    if config.strategies.ma_crossover.enabled:
        ma_rows = ma_crossover_grid(ohlcv, config)
        sensitivity["ma_crossover"] = ma_rows
        ma_flags = sensitivity_flags([row.get("sharpe_ratio") for row in ma_rows])
        all_flags.extend(f"ma_crossover: {item}" for item in ma_flags)
        _plot_ma_heatmap(ma_rows, figure_dir / "ma_crossover_sensitivity.png")
    if config.strategies.momentum.enabled:
        mom_rows = momentum_grid(ohlcv, config)
        sensitivity["momentum"] = mom_rows
        mom_flags = sensitivity_flags([row.get("sharpe_ratio") for row in mom_rows])
        all_flags.extend(f"momentum: {item}" for item in mom_flags)

    ml_notes = _read_model_reports(paths.reports, stem, diag.auc_gap_warn)
    all_flags.extend(ml_notes["flags"])

    payload = {
        "caveat": (
            "Diagnostics describe this sample under this cost model. They do not prove "
            "an edge and they do not approve live trading."
        ),
        "strategies": strategy_rows,
        "sensitivity": sensitivity,
        "model_reports": ml_notes,
        "complexity": complexity_notes(config.models),
        "flags": all_flags,
        "bootstrap_method": "moving_block_on_equity_returns",
        "permutation_method": "circular_shift_of_positions_then_costed_backtest",
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    json_path = paths.reports / f"{stem}_diagnostics.json"
    report_path = paths.reports / f"{stem}_diagnostics.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path.write_text(_markdown(payload), encoding="utf-8")
    return DiagnoseResult(
        input_path=input_path,
        report_path=report_path,
        json_path=json_path,
        n_strategies=len(strategy_rows),
        flags=all_flags,
    )


def _permutation_sharpe(
    ohlcv: pd.DataFrame,
    config: AppConfig,
    strategy: Any,
    position: np.ndarray,
    *,
    observed: float | None,
    n_perm: int,
    rng: np.random.Generator,
    periods_per_year: int,
) -> dict[str, Any]:
    if observed is None or len(position) < 8:
        return {"observed_sharpe": observed, "p_ge_observed": None, "n": 0}
    null_scores: list[float] = []
    n = len(position)
    for _ in range(n_perm):
        shift = int(rng.integers(1, n))
        rolled = pd.Series(np.roll(position, shift))
        atr = true_range_atr(ohlcv, config.strategies.atr_window)
        stop, take = stops_from_atr(
            rolled,
            ohlcv["close"],
            atr,
            config.strategies.stop_atr_multiple,
            config.strategies.take_profit_atr_multiple,
        )
        result = run_backtest(
            ohlcv,
            rolled,
            stop=stop,
            take=take,
            config=config,
            strategy_name="perm",
            realized_vol=ohlcv["close"]
            .pct_change()
            .rolling(config.risk.vol_lookback, min_periods=config.risk.vol_lookback)
            .std(),
        )
        score = sharpe_ratio(bar_returns(result.equity_curve["equity"]), periods_per_year)
        if score is not None:
            null_scores.append(score)
    if not null_scores:
        return {"observed_sharpe": observed, "p_ge_observed": None, "n": 0}
    p_ge = float(np.mean(np.array(null_scores) >= observed))
    return {
        "observed_sharpe": observed,
        "null_median_sharpe": float(np.median(null_scores)),
        "p_ge_observed": p_ge,
        "n": len(null_scores),
        "note": "Circular shifts keep the position mix but break timing. Small p is not a license to trade.",
    }


def _return_drawdown_summary(curve: pd.DataFrame) -> dict[str, Any]:
    rets = bar_returns(curve["equity"])
    dd = curve["drawdown"].astype(float).to_numpy() if "drawdown" in curve.columns else np.array([])
    def _pct(arr: np.ndarray) -> dict[str, float] | None:
        if len(arr) == 0:
            return None
        qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        return {
            "p05": float(qs[0]),
            "p25": float(qs[1]),
            "p50": float(qs[2]),
            "p75": float(qs[3]),
            "p95": float(qs[4]),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    return {"bar_returns": _pct(rets), "drawdown": _pct(dd)}


def _read_model_reports(reports: Path, stem: str, gap_warn: float) -> dict[str, Any]:
    flags: list[str] = []
    ml_path = reports / f"{stem}_ml.json"
    wf_path = reports / f"{stem}_walkforward.json"
    payload: dict[str, Any] = {"ml_json": None, "walkforward_json": None, "flags": flags}
    if ml_path.is_file():
        ml = json.loads(ml_path.read_text(encoding="utf-8"))
        payload["ml_json"] = str(ml_path)
        selected = ml.get("selected_model")
        metrics = (ml.get("metrics") or {}).get(selected) or {}
        train_auc = (metrics.get("train") or {}).get("roc_auc")
        test_auc = (metrics.get("test") or {}).get("roc_auc")
        payload["selected_model"] = selected
        payload["train_roc_auc"] = train_auc
        payload["test_roc_auc"] = test_auc
        payload["feature_importance_top"] = (ml.get("feature_importance") or {}).get(selected, [])[:8]
        flags.extend(auc_gap_flags(train_auc, test_auc, gap_warn))
    if wf_path.is_file():
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        payload["walkforward_json"] = str(wf_path)
        payload["mean_val_roc_auc"] = wf.get("mean_val_roc_auc")
        payload["mean_test_roc_auc"] = wf.get("mean_test_roc_auc")
        flags.extend(auc_gap_flags(wf.get("mean_val_roc_auc"), wf.get("mean_test_roc_auc"), gap_warn))
    payload["flags"] = flags
    return payload


def _plot_return_hist(returns: np.ndarray, path: Path, name: str) -> None:
    if len(returns) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(returns, bins=min(30, max(8, len(returns) // 5)), color="#1f4e79", alpha=0.85)
    ax.set_title(f"{name} equity bar returns (costed)")
    ax.set_xlabel("return")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_ma_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="fast", columns="slow", values="sharpe_ratio")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
    ax.set_xlabel("slow SMA")
    ax.set_ylabel("fast SMA")
    ax.set_title("MA crossover Sharpe vs nearby parameters")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _jsonable(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"[{_fmt(value[0])}, {_fmt(value[1])}]"
    return str(value)


def _markdown(payload: dict[str, Any]) -> str:
    lines = []
    for row in payload["strategies"]:
        m = row["metrics"]
        b = row["bootstrap"]
        p = row["permutation"]
        flag_txt = "; ".join(row["flags"]) if row["flags"] else "none"
        lines.append(
            f"| `{row['strategy']}` | {m.get('number_of_trades')} | {_fmt(m.get('total_return'))} | "
            f"{_fmt(m.get('max_drawdown'))} | {_fmt(m.get('sharpe_ratio'))} | "
            f"{_fmt(b.get('mean_ci'))} | {_fmt(p.get('p_ge_observed'))} | {flag_txt} |"
        )
    ma = payload.get("sensitivity", {}).get("ma_crossover") or []
    ma_lines = [
        f"| {row['fast']} | {row['slow']} | {_fmt(row.get('sharpe_ratio'))} | "
        f"{_fmt(row.get('total_return'))} | {row.get('is_configured')} |"
        for row in ma
    ]
    flags = payload.get("flags") or []
    flag_block = "\n".join(f"- {item}" for item in flags) or "- none"
    ml = payload.get("model_reports") or {}
    return f"""# Diagnostics report (Phase 9)

{payload['caveat']}

Block bootstrap uses **equity bar returns** (time-series blocks). The permutation test **circular-shifts positions** and re-runs the costed engine. Neither is a live result.

| Strategy | Trades | Total return | Max DD | Sharpe | Mean-return CI | Permutation p(Sharpe ≥ obs) | Flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(lines)}

## Parameter sensitivity (MA crossover)

Nearby windows should not collapse to one magical pair.

| Fast | Slow | Sharpe | Total return | Configured? |
| --- | --- | --- | --- | --- |
{chr(10).join(ma_lines) or '| (disabled) | | | | |'}

## Model-side checks

Train vs test ROC-AUC (if `train` / `walk-forward` reports exist): val/test gap warnings are listed below. Feature importances are from the train fold only.

Selected model: `{ml.get('selected_model')}`  
Train AUC: {_fmt(ml.get('train_roc_auc'))} | Test AUC: {_fmt(ml.get('test_roc_auc'))}  
Walk-forward mean val/test AUC: {_fmt(ml.get('mean_val_roc_auc'))} / {_fmt(ml.get('mean_test_roc_auc'))}

Complexity: `{payload.get('complexity')}`

## Flags

{flag_block}

If a backtest looks too good to be true, treat it as a bug in the research design until proven otherwise.
"""
