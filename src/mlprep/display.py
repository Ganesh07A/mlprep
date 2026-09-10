"""Dataset profile display.

Provides two output modes:
  - Rich terminal tables (default)
  - JSON (for automation and CI pipelines)
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table

from mlprep.profiler import profile_dataset

console = Console()


def display_profile(df: pd.DataFrame, fmt: str = "rich") -> None:
    """Display a dataset profile in the chosen format.

    Args:
        df: DataFrame to profile.
        fmt: Output format — ``"rich"`` (default) or ``"json"``.
    """
    if fmt == "json":
        display_profile_json(df)
    else:
        display_profile_rich(df)


def display_profile_rich(df: pd.DataFrame) -> None:
    """Display a dataset profile as Rich terminal tables."""
    profile = profile_dataset(df)

    # ── Missing values table ───────────────────────────────────────────────
    missing_table = Table(title="MISSING VALUES")
    missing_table.add_column("Column")
    missing_table.add_column("Missing")
    missing_table.add_column("Percentage")

    if profile.missing_count.empty:
        missing_table.add_row("[dim]No missing values[/dim]", "", "")
    else:
        for column in profile.missing_count.index:
            missing_table.add_row(
                column,
                str(profile.missing_count[column]),
                f"{profile.missing_percentage[column]:.1f}%",
            )

    console.print(missing_table)

    # ── Data types table ───────────────────────────────────────────────────
    types_table = Table(title="DATA TYPES")
    types_table.add_column("Column")
    types_table.add_column("Type")

    for column in profile.data_types.index:
        if column in profile.numerical_values:
            col_type = "numerical"
        elif column in profile.categorical_values:
            col_type = "categorical"
        else:
            col_type = str(profile.data_types[column])

        types_table.add_row(column, col_type)

    console.print(types_table)

    # ── Summary ────────────────────────────────────────────────────────────
    console.print(
        f"\n[dim]Rows: {profile.rows:,}  |  "
        f"Columns: {profile.columns}  |  "
        f"Numerical: {len(profile.numerical_values)}  |  "
        f"Categorical: {len(profile.categorical_values)}[/dim]"
    )


def display_profile_json(df: pd.DataFrame) -> None:
    """Emit a dataset profile as JSON to stdout.

    Designed for automation, CI pipelines, and scripting.

    Output schema::

        {
          "rows": int,
          "columns": int,
          "numerical_columns": [str, ...],
          "categorical_columns": [str, ...],
          "missing": {column: {"count": int, "percentage": float}, ...}
        }
    """
    profile = profile_dataset(df)

    missing: dict = {}
    for col in profile.missing_count.index:
        missing[col] = {
            "count": int(profile.missing_count[col]),
            "percentage": round(float(profile.missing_percentage[col]), 2),
        }

    output = {
        "rows": profile.rows,
        "columns": profile.columns,
        "numerical_columns": profile.numerical_values,
        "categorical_columns": profile.categorical_values,
        "missing": missing,
    }

    sys.stdout.write(json.dumps(output, indent=2) + "\n")
