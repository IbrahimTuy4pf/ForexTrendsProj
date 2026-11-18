"""CLI for data ingestion and exploration."""

from __future__ import annotations

import argparse
import logging

from quant_forex.config.loader import load_config
from quant_forex.config.schema import AppConfig
from quant_forex.data.pipeline import run_ingestion_pipeline
from quant_forex.exploration.pipeline import run_exploration
from quant_forex.features.pipeline import run_feature_pipeline
from quant_forex.strategies.pipeline import run_baselines
from quant_forex.backtesting.pipeline import run_backtest_pipeline
from quant_forex.utils.logging import configure_logging
from quant_forex.utils.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantitative forex research platform (Phase 11: research dashboard)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Download/load, validate, and store OHLCV data.")
    _add_shared_io_args(ingest)

    explore = sub.add_parser(
        "explore",
        help="Descriptive charts and statistics for a processed OHLCV file.",
    )
    _add_shared_io_args(explore)
    explore.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )

    features = sub.add_parser(
        "features",
        help="Build causal technical/time features from processed OHLCV.",
    )
    _add_shared_io_args(features)
    features.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )

    baselines = sub.add_parser(
        "baselines",
        help="Run rule-based baseline strategies (signals only; not a costed backtest).",
    )
    _add_shared_io_args(baselines)
    baselines.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )

    backtest = sub.add_parser(
        "backtest",
        help="Run a costed next-open backtest of enabled baseline strategies.",
    )
    _add_shared_io_args(backtest)
    backtest.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )
    backtest.add_argument(
        "--strategy",
        default=None,
        help="Optional single strategy name (e.g. ma_crossover).",
    )

    preview = sub.add_parser(
        "risk-preview",
        help="Show conservative position size at initial capital (not an order).",
    )
    _add_shared_io_args(preview)

    train = sub.add_parser(
        "train",
        help="Fit simple classifiers on chronological folds (selection uses validation only).",
    )
    _add_shared_io_args(train)
    train.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )

    walk = sub.add_parser(
        "walk-forward",
        help="Refit classifiers on expanding/rolling folds; score only the purged OOS window.",
    )
    _add_shared_io_args(walk)
    walk.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )

    diagnose = sub.add_parser(
        "diagnose",
        help="Bootstrap CIs, permutation tests, parameter sensitivity, overfitting flags.",
    )
    _add_shared_io_args(diagnose)
    diagnose.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )

    paper = sub.add_parser(
        "paper",
        help="Replay a simulated paper book (opt-in; never sends broker orders).",
    )
    _add_shared_io_args(paper)
    paper.add_argument(
        "--input",
        default=None,
        help="Processed CSV path. Default: newest *_processed.csv in data/processed.",
    )
    paper.add_argument(
        "--strategy",
        default=None,
        help="Strategy name (default: paper.strategy or first enabled baseline).",
    )
    paper.add_argument(
        "--enable-paper",
        action="store_true",
        help="Set execution.paper_trading true for this run. Does not enable live trading.",
    )

    dash = sub.add_parser(
        "dashboard",
        help="Open the read-only Streamlit research dashboard (no broker orders).",
    )
    _add_shared_io_args(dash)
    return parser


def _add_shared_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--source",
        choices=["csv", "yfinance"],
        default=None,
        help="Override data.source from the config file.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Override data.csv_path when using a CSV source.",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Override instrument.symbol (e.g. EURUSD, GBPUSD).",
    )


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if getattr(args, "source", None):
        config.data.source = args.source
    if getattr(args, "csv_path", None):
        config.data.csv_path = args.csv_path
    if getattr(args, "symbol", None):
        config.instrument.symbol = args.symbol
    if getattr(args, "enable_paper", False):
        config.execution.paper_trading = True
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    paths = ProjectPaths.find()
    config = _apply_overrides(load_config(paths.resolve(args.config)), args)
    configure_logging(config.project.log_level)
    logger = logging.getLogger("quant_forex")

    if args.command == "ingest":
        result = run_ingestion_pipeline(config, paths)
        logger.info("Ingestion complete. dataset_id=%s", result.dataset_id)
        print(f"dataset_id={result.dataset_id}")
        print(f"quality_report={result.quality_report_path}")
        print(f"processed_rows={result.n_processed_rows}")
        print(f"sqlite={paths.resolve(config.storage.sqlite_path)}")
        return 0

    if args.command == "explore":
        result = run_exploration(config, paths, input_csv=args.input)
        logger.info("Exploration complete. report=%s", result.report_path)
        print(f"input={result.input_path}")
        print(f"exploration_report={result.report_path}")
        print(f"figures={result.figure_dir}")
        print(f"rows={result.n_rows}")
        print(f"sessions_applicable={result.summary.sessions_applicable}")
        return 0

    if args.command == "features":
        result = run_feature_pipeline(config, paths, input_csv=args.input)
        logger.info("Features written to %s", result.output_csv)
        print(f"input={result.input_path}")
        print(f"features_csv={result.output_csv}")
        print(f"feature_report={result.report_path}")
        print(f"rows={result.n_rows}")
        print(f"n_features={result.n_features}")
        return 0

    if args.command == "baselines":
        result = run_baselines(config, paths, input_csv=args.input)
        logger.info("Baseline report %s", result.report_path)
        print(f"input={result.input_path}")
        print(f"signals_csv={result.signals_path}")
        print(f"positions_csv={result.positions_path}")
        print(f"baseline_report={result.report_path}")
        print("strategies=" + ",".join(item.name for item in result.runs))
        return 0

    if args.command == "backtest":
        result = run_backtest_pipeline(
            config, paths, input_csv=args.input, strategy_name=args.strategy
        )
        logger.info("Backtest report %s", result.report_path)
        print(f"input={result.input_path}")
        print(f"backtest_report={result.report_path}")
        print(f"trades_csv={result.trades_path}")
        print(f"equity_csv={result.equity_path}")
        print("strategies=" + ",".join(item.strategy for item in result.results))
        return 0

    if args.command == "risk-preview":
        from quant_forex.risk.preview import preview_size

        decision = preview_size(config, paths)
        print(f"method={decision.method}")
        print(f"units={decision.units:.4f}")
        print(f"notional={decision.notional:.2f}")
        print(f"leverage={decision.leverage:.4f}")
        print(f"cash_at_risk={decision.cash_at_risk:.2f}")
        print(f"capped_by={','.join(decision.capped_by) if decision.capped_by else 'none'}")
        print(f"allowed={decision.allowed}")
        print(f"reason={decision.reason}")
        return 0

    if args.command == "train":
        from quant_forex.models.pipeline import run_ml_pipeline

        result = run_ml_pipeline(config, paths, input_csv=args.input)
        logger.info("ML report %s selected=%s", result.report_path, result.selected_model)
        print(f"input={result.input_path}")
        print(f"ml_report={result.report_path}")
        print(f"ml_json={result.json_path}")
        print(f"predictions_csv={result.predictions_path}")
        print(f"artifact={result.artifact_path}")
        print(f"selected_model={result.selected_model}")
        print(f"split_method={result.split_method}")
        return 0

    if args.command == "walk-forward":
        from quant_forex.evaluation.pipeline import run_walk_forward

        result = run_walk_forward(config, paths, input_csv=args.input)
        logger.info("Walk-forward report %s folds=%s", result.report_path, result.n_folds)
        print(f"input={result.input_path}")
        print(f"walkforward_report={result.report_path}")
        print(f"walkforward_json={result.json_path}")
        print(f"predictions_csv={result.predictions_path}")
        print(f"n_folds={result.n_folds}")
        print(f"mode={result.mode}")
        return 0

    if args.command == "diagnose":
        from quant_forex.evaluation.diagnose import run_diagnostics

        result = run_diagnostics(config, paths, input_csv=args.input)
        logger.info("Diagnostics report %s", result.report_path)
        print(f"input={result.input_path}")
        print(f"diagnostics_report={result.report_path}")
        print(f"diagnostics_json={result.json_path}")
        print(f"n_strategies={result.n_strategies}")
        print(f"n_flags={len(result.flags)}")
        return 0

    if args.command == "paper":
        from quant_forex.execution.pipeline import run_paper

        result = run_paper(
            config, paths, input_csv=args.input, strategy_name=args.strategy
        )
        logger.info("Paper session %s", result.report_path)
        print(f"input={result.input_path}")
        print(f"paper_report={result.report_path}")
        print(f"paper_json={result.json_path}")
        print(f"ledger={result.ledger_path}")
        print(f"strategy={result.strategy}")
        print(f"equity={result.equity:.4f}")
        print(f"open_units={result.units:.4f}")
        print(f"closed_trades={result.n_trades}")
        print("venue=paper_simulator")
        return 0

    if args.command == "dashboard":
        from quant_forex.dashboard.launch import launch_dashboard

        logger.info("Starting research dashboard (read-only).")
        return launch_dashboard()

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
