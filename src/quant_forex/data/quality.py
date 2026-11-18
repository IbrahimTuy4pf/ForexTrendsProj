"""Data-quality reporting. Validation never silently mutates the input frame."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from quant_forex.data.schema import OHLCV_COLUMNS

Severity = Literal["info", "warning", "error"]


class QualityIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    count: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    symbol: str
    timeframe: str
    source: str
    n_rows: int
    n_columns: int
    date_range_start: str | None
    date_range_end: str | None
    timezone: str
    inferred_frequency: str | None
    expected_frequency: str | None
    median_bar_seconds: float | None
    missing_values: dict[str, int]
    duplicate_timestamps: int
    duplicate_rows: int
    n_abnormal_ohlc: int
    n_non_positive_prices: int
    n_abnormal_returns: int
    issues: list[QualityIssue]
    transformations: list[dict[str, Any]] = Field(default_factory=list)
    generated_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def _timestamp_series(frame: pd.DataFrame) -> pd.Series | None:
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex):
        idx = frame.index
        if idx.tz is None:
            return pd.Series(idx.tz_localize("UTC"))
        return pd.Series(idx.tz_convert("UTC"))
    return None


def inspect_ohlc_consistency(frame: pd.DataFrame) -> pd.Series:
    """True where OHLC geometry is impossible."""
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    close = frame["close"]
    return (
        (high < low)
        | (high < open_)
        | (high < close)
        | (low > open_)
        | (low > close)
    )


def build_quality_report(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    source: str,
    timezone: str,
    expected_frequency: str | None,
    abnormal_return_threshold: float,
    transformations: list[dict[str, Any]] | None = None,
) -> DataQualityReport:
    issues: list[QualityIssue] = []
    working = frame.copy()

    required = [col for col in OHLCV_COLUMNS if col != "volume"]
    missing_cols = [col for col in required if col not in working.columns]
    if missing_cols:
        issues.append(
            QualityIssue(
                code="missing_columns",
                severity="error",
                message=f"Required columns missing: {missing_cols}",
                count=len(missing_cols),
                details={"columns": missing_cols},
            )
        )

    stamps = _timestamp_series(working)

    n_unparsed = 0
    date_start = date_end = None
    inferred_frequency = None
    median_bar_seconds = None
    duplicate_timestamps = 0
    n_abnormal_returns = 0

    if stamps is None:
        issues.append(
            QualityIssue(
                code="missing_timestamp",
                severity="error",
                message="No timestamp column or DatetimeIndex found.",
            )
        )
    else:
        n_unparsed = int(stamps.isna().sum())
        if n_unparsed:
            issues.append(
                QualityIssue(
                    code="unparsed_timestamps",
                    severity="error",
                    message="Some timestamps could not be parsed.",
                    count=n_unparsed,
                )
            )
        valid = stamps.dropna()
        if not valid.empty:
            date_start = valid.min().isoformat()
            date_end = valid.max().isoformat()
            ordered = valid.sort_values()
            deltas = ordered.diff().dropna()
            if not deltas.empty:
                median_bar_seconds = float(deltas.median().total_seconds())
            try:
                inferred_frequency = pd.infer_freq(pd.DatetimeIndex(ordered.sort_values()))
            except (ValueError, TypeError):
                inferred_frequency = None
            duplicate_timestamps = int(valid.duplicated().sum())
            if duplicate_timestamps:
                issues.append(
                    QualityIssue(
                        code="duplicate_timestamps",
                        severity="warning",
                        message="Duplicate timestamps detected.",
                        count=duplicate_timestamps,
                    )
                )
            if not valid.is_monotonic_increasing:
                issues.append(
                    QualityIssue(
                        code="not_sorted",
                        severity="warning",
                        message="Rows are not strictly increasing in time.",
                    )
                )

    duplicate_rows = int(working.duplicated().sum())
    if duplicate_rows:
        issues.append(
            QualityIssue(
                code="duplicate_rows",
                severity="warning",
                message="Exact duplicate rows detected.",
                count=duplicate_rows,
            )
        )

    missing_values: dict[str, int] = {}
    for column in working.columns:
        missing_values[str(column)] = int(working[column].isna().sum())
    total_missing = sum(missing_values.values())
    if total_missing:
        issues.append(
            QualityIssue(
                code="missing_values",
                severity="warning",
                message="Missing values present; they are not filled unless a policy is applied.",
                count=total_missing,
                details=missing_values,
            )
        )

    n_abnormal_ohlc = 0
    n_non_positive = 0
    price_cols = [c for c in ("open", "high", "low", "close") if c in working.columns]
    if len(price_cols) == 4:
        numeric = working[price_cols].apply(pd.to_numeric, errors="coerce")
        n_non_positive = int((numeric <= 0).any(axis=1).sum())
        if n_non_positive:
            issues.append(
                QualityIssue(
                    code="non_positive_prices",
                    severity="error",
                    message="Non-positive prices found.",
                    count=n_non_positive,
                )
            )
        n_abnormal_ohlc = int(inspect_ohlc_consistency(numeric).sum())
        if n_abnormal_ohlc:
            issues.append(
                QualityIssue(
                    code="inconsistent_ohlc",
                    severity="error",
                    message="OHLC geometry is inconsistent (e.g. high < low).",
                    count=n_abnormal_ohlc,
                )
            )
        if "close" in numeric.columns:
            returns = numeric["close"].pct_change()
            n_abnormal_returns = int((returns.abs() > abnormal_return_threshold).sum())
            if n_abnormal_returns:
                issues.append(
                    QualityIssue(
                        code="abnormal_returns",
                        severity="warning",
                        message=(
                            f"Close-to-close returns exceed {abnormal_return_threshold:.2%}."
                        ),
                        count=n_abnormal_returns,
                        details={"threshold": abnormal_return_threshold},
                    )
                )

    if (
        expected_frequency
        and inferred_frequency
        and inferred_frequency.replace(" ", "") != expected_frequency.replace(" ", "")
    ):
        issues.append(
            QualityIssue(
                code="frequency_mismatch",
                severity="info",
                message=(
                    f"Inferred frequency {inferred_frequency!r} differs from expected "
                    f"{expected_frequency!r}."
                ),
            )
        )

    if "volume" in working.columns:
        vol_missing = int(working["volume"].isna().sum())
        if vol_missing == len(working) or (
            working["volume"].fillna(0).eq(0).all() if len(working) else False
        ):
            issues.append(
                QualityIssue(
                    code="empty_volume",
                    severity="info",
                    message="Volume is missing or zero for all rows (common for FX sources).",
                    count=len(working),
                )
            )

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        n_rows=int(len(working)),
        n_columns=int(working.shape[1]),
        date_range_start=date_start,
        date_range_end=date_end,
        timezone=timezone,
        inferred_frequency=inferred_frequency,
        expected_frequency=expected_frequency,
        median_bar_seconds=median_bar_seconds,
        missing_values=missing_values,
        duplicate_timestamps=duplicate_timestamps,
        duplicate_rows=duplicate_rows,
        n_abnormal_ohlc=n_abnormal_ohlc,
        n_non_positive_prices=n_non_positive,
        n_abnormal_returns=n_abnormal_returns,
        issues=issues,
        transformations=transformations or [],
    )


def report_to_markdown(report: DataQualityReport) -> str:
    issue_lines = [
        f"- **{issue.severity.upper()}** `{issue.code}`: {issue.message} (count={issue.count})"
        for issue in report.issues
    ] or ["- None"]
    transform_lines = [
        f"- `{item.get('name')}`: {item.get('description')} "
        f"(rows {item.get('rows_before')} → {item.get('rows_after')})"
        for item in report.transformations
    ] or ["- None"]
    missing = ", ".join(f"{k}={v}" for k, v in report.missing_values.items()) or "none"
    return f"""# Data quality report

Generated (UTC): {report.generated_at_utc.isoformat()}

| Field | Value |
| --- | --- |
| Symbol | `{report.symbol}` |
| Timeframe | `{report.timeframe}` |
| Source | `{report.source}` |
| Rows | {report.n_rows} |
| Columns | {report.n_columns} |
| Date start | {report.date_range_start} |
| Date end | {report.date_range_end} |
| Timezone (declared) | {report.timezone} |
| Inferred frequency | {report.inferred_frequency} |
| Expected frequency | {report.expected_frequency} |
| Median bar seconds | {report.median_bar_seconds} |
| Duplicate timestamps | {report.duplicate_timestamps} |
| Duplicate rows | {report.duplicate_rows} |
| Abnormal OHLC rows | {report.n_abnormal_ohlc} |
| Non-positive price rows | {report.n_non_positive_prices} |
| Abnormal returns | {report.n_abnormal_returns} |

## Missing values

{missing}

## Issues

{chr(10).join(issue_lines)}

## Logged transformations

{chr(10).join(transform_lines)}

This report does not imply that the series is tradable or that any strategy is profitable.
"""


def empty_like_quality_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OHLCV_COLUMNS))
