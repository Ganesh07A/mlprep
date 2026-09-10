"""Data quality checks and recommendations.

Checks datasets for common ML anti-patterns and returns structured warnings.
Never silently modifies data — only warns and recommends.

Architecture
────────────
    DataFrame
        ↓
    run_quality_checks(df)
        ↓
    list[QualityWarning]
        ↓
    display_quality_warnings(warnings)   ← Rich terminal output

Usage::

    from mlprep.quality import run_quality_checks, display_quality_warnings

    warnings = run_quality_checks(df)
    if warnings:
        display_quality_warnings(warnings)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from rich.console import Console
from rich.table import Table

from mlprep.logging_config import get_logger

console = Console()
log = get_logger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

MISSING_HIGH_THRESHOLD = 0.50          # > 50% missing → warn
CARDINALITY_HIGH_THRESHOLD = 50        # > 50 unique values in a categorical col
IMBALANCE_THRESHOLD = 0.95             # dominant class > 95% of rows
IDENTIFIER_UNIQUENESS_THRESHOLD = 0.95 # > 95% unique values → likely an ID
_ID_PATTERNS = re.compile(r"(^|_)(id|key|uuid|pk|index|num|no|code)($|_)", re.IGNORECASE)


# ── Safe column accessor ──────────────────────────────────────────────────────


def _get_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the first occurrence of *col* as a Series, even if duplicated."""
    series = df[col]
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0]
    return series


# ── Warning model ─────────────────────────────────────────────────────────────


@dataclass
class QualityWarning:
    """A single data quality issue detected in a dataset."""

    column: str | None   # None for dataset-level checks
    check_name: str
    message: str
    recommendation: str
    severity: str = "warning"  # "warning" | "info"


# ── Individual checks ─────────────────────────────────────────────────────────


def _check_high_missingness(df: pd.DataFrame) -> list[QualityWarning]:
    warnings = []
    for col in dict.fromkeys(df.columns):
        rate = _get_col(df, col).isnull().mean()
        if rate > MISSING_HIGH_THRESHOLD:
            warnings.append(
                QualityWarning(
                    column=col,
                    check_name="high_missingness",
                    message=f"{col}: {rate:.0%} of values are missing.",
                    recommendation=(
                        "Consider whether this column carries enough signal. "
                        "High missingness can degrade model performance."
                    ),
                )
            )
    return warnings


def _check_constant_columns(df: pd.DataFrame) -> list[QualityWarning]:
    warnings = []
    for col in dict.fromkeys(df.columns):  # iterate unique col names to avoid dup-col ambiguity
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]  # take first if duplicated
        if series.nunique(dropna=True) <= 1:
            warnings.append(
                QualityWarning(
                    column=col,
                    check_name="constant_column",
                    message=f"{col}: has only one unique value (constant column).",
                    recommendation=(
                        "Constant columns provide no information to a model. "
                        "Consider excluding this column."
                    ),
                )
            )
    return warnings


def _check_duplicate_rows(df: pd.DataFrame) -> list[QualityWarning]:
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        pct = n_dupes / len(df)
        return [
            QualityWarning(
                column=None,
                check_name="duplicate_rows",
                message=f"Dataset has {n_dupes:,} duplicate row(s) ({pct:.1%} of total).",
                recommendation=(
                    "Duplicate rows can bias training. "
                    "Review whether they are legitimate or data collection artifacts."
                ),
            )
        ]
    return []


def _check_duplicate_column_names(df: pd.DataFrame) -> list[QualityWarning]:
    seen: set[str] = set()
    dupes: list[str] = []
    for col in df.columns:
        if col in seen:
            dupes.append(col)
        seen.add(col)
    if dupes:
        return [
            QualityWarning(
                column=None,
                check_name="duplicate_column_names",
                message=f"Duplicate column names detected: {dupes}.",
                recommendation=(
                    "Rename duplicate columns before preprocessing. "
                    "MLPrep may behave unexpectedly with duplicate names."
                ),
                severity="warning",
            )
        ]
    return []


def _check_high_cardinality(df: pd.DataFrame) -> list[QualityWarning]:
    warnings = []
    cat_cols = df.select_dtypes(include=["str", "object", "category"]).columns
    for col in cat_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > CARDINALITY_HIGH_THRESHOLD:
            warnings.append(
                QualityWarning(
                    column=col,
                    check_name="high_cardinality",
                    message=(
                        f"{col}: {n_unique} unique values "
                        f"(high cardinality for a categorical column)."
                    ),
                    recommendation=(
                        "One-hot encoding high-cardinality columns creates many features "
                        "and can hurt model performance. "
                        "Consider target encoding, frequency encoding, or exclusion."
                    ),
                )
            )
    return warnings


def _check_identifier_columns(df: pd.DataFrame) -> list[QualityWarning]:
    warnings = []
    n_rows = max(len(df), 1)
    for col in dict.fromkeys(df.columns):  # unique col names
        uniqueness = _get_col(df, col).nunique(dropna=True) / n_rows
        name_looks_like_id = bool(_ID_PATTERNS.search(col))

        is_suspicious = (
            (name_looks_like_id and uniqueness >= IDENTIFIER_UNIQUENESS_THRESHOLD)
            or (uniqueness >= 0.99 and n_rows >= 10)
        )
        if is_suspicious:
            warnings.append(
                QualityWarning(
                    column=col,
                    check_name="identifier_column",
                    message=(
                        f"{col}: looks like an identifier "
                        f"({uniqueness:.0%} unique values, id-like name: {name_looks_like_id})."
                    ),
                    recommendation=(
                        "Identifier columns should not be used as ML features. "
                        "Consider excluding this column before preprocessing."
                    ),
                )
            )
    return warnings


def _check_potential_datetime(df: pd.DataFrame) -> list[QualityWarning]:
    warnings = []
    object_cols = df.select_dtypes(include=["str", "object"]).columns
    for col in object_cols:
        sample = df[col].dropna().head(20)
        if sample.empty:
            continue
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:
                warnings.append(
                    QualityWarning(
                        column=col,
                        check_name="potential_datetime",
                        message=f"{col}: values appear to be dates/timestamps.",
                        recommendation=(
                            "MLPrep does not extract datetime features automatically. "
                            "Consider parsing and engineering features (year, month, day) "
                            "before preprocessing."
                        ),
                        severity="info",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("Datetime parse failed for %s: %s", col, exc)
    return warnings


def _check_imbalanced_categorical(df: pd.DataFrame) -> list[QualityWarning]:
    warnings = []
    cat_cols = df.select_dtypes(include=["str", "object", "category"]).columns
    for col in cat_cols:
        if df[col].empty:
            continue
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        if top_freq > IMBALANCE_THRESHOLD:
            top_val = df[col].value_counts().index[0]
            warnings.append(
                QualityWarning(
                    column=col,
                    check_name="imbalanced_categorical",
                    message=(
                        f"{col}: '{top_val}' accounts for {top_freq:.0%} of values "
                        "(highly imbalanced)."
                    ),
                    recommendation=(
                        "Highly imbalanced categoricals produce near-zero-variance features "
                        "after one-hot encoding, which rarely helps models."
                    ),
                    severity="info",
                )
            )
    return warnings


# ── Public API ────────────────────────────────────────────────────────────────


def run_quality_checks(df: pd.DataFrame) -> list[QualityWarning]:
    """Run all data quality checks on *df* and return structured warnings.

    Results are warnings only — no data is modified.

    Args:
        df: DataFrame to check.

    Returns:
        List of ``QualityWarning`` objects. Empty list means no issues found.
    """
    checks = [
        _check_duplicate_column_names,
        _check_duplicate_rows,
        _check_constant_columns,
        _check_high_missingness,
        _check_identifier_columns,
        _check_high_cardinality,
        _check_imbalanced_categorical,
        _check_potential_datetime,
    ]
    warnings: list[QualityWarning] = []
    for check in checks:
        warnings.extend(check(df))
    return warnings


def display_quality_warnings(warnings: list[QualityWarning]) -> None:
    """Display data quality warnings as a Rich table.

    Args:
        warnings: List of ``QualityWarning`` objects from ``run_quality_checks``.
    """
    if not warnings:
        return

    table = Table(title="⚠  Data Quality Warnings", show_header=True, show_lines=True)
    table.add_column("Column", style="bold", no_wrap=True)
    table.add_column("Check")
    table.add_column("Issue")
    table.add_column("Recommendation", max_width=55)

    for w in warnings:
        severity_color = "yellow" if w.severity == "warning" else "cyan"
        table.add_row(
            w.column or "[dim](dataset)[/dim]",
            f"[{severity_color}]{w.check_name}[/{severity_color}]",
            w.message,
            w.recommendation,
        )

    console.print()
    console.print(table)
    console.print(
        f"[yellow]{len(warnings)} quality warning(s) found. "
        "Review before proceeding.[/yellow]\n"
    )
