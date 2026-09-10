"""Preprocessing report.

The report receives ``PreprocessConfig`` + ``DatasetProfile`` + output paths and
generates a trustworthy account of what happened — no guessing from inspecting
the fitted pipeline after the fact.

Usage::

    report = build_report(
        config=config,
        profile=profile,
        input_path=file_path,
        output_path=output,
        pipeline_path=pipeline_output,
    )
    display_report(report)
    save_report_json(report, output.parent / "report.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mlprep.config import PreprocessConfig
from mlprep.models import datasetProfile

console = Console()


# ── Report model ──────────────────────────────────────────────────────────────


@dataclass
class ColumnTransformation:
    """Records what transformations were applied to a single column."""

    column: str
    column_type: str  # "numerical" | "categorical"
    imputation: str
    transformation: str  # "standard scaling" | "one-hot encoding"

    def describe(self) -> str:
        return f"{self.imputation} → {self.transformation}"


@dataclass
class PreprocessReport:
    """Complete record of a preprocessing run."""

    input_file: str
    rows: int
    numerical_columns: list[str]
    categorical_columns: list[str]
    transformations: list[ColumnTransformation]
    output_file: str
    pipeline_file: str
    report_file: str


# ── Builder ───────────────────────────────────────────────────────────────────


def build_report(
    config: PreprocessConfig,
    profile: datasetProfile,
    input_path: Path,
    output_path: Path,
    pipeline_path: Path,
    report_path: Path,
) -> PreprocessReport:
    """Build a ``PreprocessReport`` from config and profile data.

    The report is driven entirely by ``config`` — it never inspects the fitted
    pipeline — so it always accurately reflects the user's intent.
    """
    transformations: list[ColumnTransformation] = []

    for col in profile.numerical_values:
        imputation = _describe_numerical_imputation(config)
        transformations.append(
            ColumnTransformation(
                column=col,
                column_type="numerical",
                imputation=imputation,
                transformation="standard scaling",
            )
        )

    for col in profile.categorical_values:
        imputation = _describe_categorical_imputation(config)
        transformations.append(
            ColumnTransformation(
                column=col,
                column_type="categorical",
                imputation=imputation,
                transformation="one-hot encoding",
            )
        )

    return PreprocessReport(
        input_file=str(input_path),
        rows=profile.rows,
        numerical_columns=profile.numerical_values,
        categorical_columns=profile.categorical_values,
        transformations=transformations,
        output_file=str(output_path),
        pipeline_file=str(pipeline_path),
        report_file=str(report_path),
    )


def _describe_numerical_imputation(config: PreprocessConfig) -> str:
    if config.numerical_strategy == "constant":
        return f"constant imputation (fill={config.resolved_numerical_fill})"
    return f"{config.numerical_strategy} imputation"


def _describe_categorical_imputation(config: PreprocessConfig) -> str:
    if config.categorical_strategy == "constant":
        return f"constant imputation (fill='{config.resolved_categorical_fill}')"
    return f"{config.categorical_strategy} imputation"


# ── Display ───────────────────────────────────────────────────────────────────


def display_report(report: PreprocessReport) -> None:
    """Print the preprocessing report to the terminal using Rich."""
    console.print(Panel("[bold]Preprocessing Report[/bold]", expand=False))

    # Summary
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Key", style="bold cyan")
    summary.add_column("Value")
    summary.add_row("Input", report.input_file)
    summary.add_row("Rows", f"{report.rows:,}")
    summary.add_row("Numerical columns", str(len(report.numerical_columns)))
    summary.add_row("Categorical columns", str(len(report.categorical_columns)))
    console.print(summary)

    # Transformations per column
    if report.transformations:
        console.print()
        t = Table(title="Transformations", show_header=True)
        t.add_column("Column", style="bold")
        t.add_column("Type")
        t.add_column("Applied")
        for tr in report.transformations:
            t.add_row(tr.column, tr.column_type, tr.describe())
        console.print(t)

    # Outputs
    console.print()
    out = Table(show_header=False, box=None, padding=(0, 2))
    out.add_column("Key", style="bold cyan")
    out.add_column("Value", style="green")
    out.add_row("Output dataset", report.output_file)
    out.add_row("Pipeline", report.pipeline_file)
    out.add_row("Report", report.report_file)
    console.print(out)


# ── JSON export ───────────────────────────────────────────────────────────────


def save_report_json(report: PreprocessReport, path: Path) -> None:
    """Save the preprocessing report as a machine-readable JSON file."""
    data = asdict(report)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
