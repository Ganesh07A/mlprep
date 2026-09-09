"""Interactive preprocessing wizard.

When the terminal is interactive (``sys.stdin.isatty()``), the wizard prompts
the user for each preprocessing choice using ``questionary``.

When running in a non-interactive environment (CI, pytest CliRunner, pipes),
it silently returns sensible defaults so the CLI still works.

Entry point
───────────
Call ``build_config_interactive()`` to run the full wizard and receive a
``PreprocessConfig`` that can be passed directly to the pipeline engine.
"""

from __future__ import annotations

import sys

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mlprep.config import (
    CATEGORICAL_STRATEGIES,
    NUMERICAL_STRATEGIES,
    PreprocessConfig,
    default_config,
)

console = Console()


# ── TTY detection ─────────────────────────────────────────────────────────────


def _is_interactive() -> bool:
    """Return True when running in a real interactive terminal."""
    return sys.stdin.isatty()


# ── Individual prompts ────────────────────────────────────────────────────────


def ask_numerical_strategy() -> str:
    """Prompt for numerical missing-value strategy, or return default."""
    if not _is_interactive():
        return "median"
    result = questionary.select(
        "How should numerical missing values be handled?",
        choices=list(NUMERICAL_STRATEGIES),
        default="median",
    ).ask()
    return result if result is not None else "median"


def ask_categorical_strategy() -> str:
    """Prompt for categorical missing-value strategy, or return default."""
    if not _is_interactive():
        return "most_frequent"
    result = questionary.select(
        "How should categorical missing values be handled?",
        choices=list(CATEGORICAL_STRATEGIES),
        default="most_frequent",
    ).ask()
    return result if result is not None else "most_frequent"


def ask_confirmation(config: PreprocessConfig) -> bool:
    """Show a review screen and ask whether to proceed.

    Returns True if the user confirms, False if they cancel.
    In non-interactive mode always returns True.
    """
    if not _is_interactive():
        return True

    _display_selected_config(config)

    result = questionary.confirm(
        "Apply preprocessing with these settings?",
        default=True,
    ).ask()
    return result if result is not None else True


# ── Config display ────────────────────────────────────────────────────────────


def _display_selected_config(config: PreprocessConfig) -> None:
    """Print the selected preprocessing configuration as a Rich table."""
    table = Table(title="Preprocessing Configuration", show_header=True)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="green")

    table.add_row("Numerical strategy", config.numerical_strategy)
    table.add_row("Categorical strategy", config.categorical_strategy)
    table.add_row("Encoding", f"✓ One-Hot Encoding ({config.encoding})")
    table.add_row("Scaling", f"✓ Standard Scaling ({config.scaling})")

    console.print(table)


# ── Full wizard ───────────────────────────────────────────────────────────────


def build_config_interactive() -> PreprocessConfig | None:
    """Run the full interactive wizard and return a ``PreprocessConfig``.

    Returns:
        A ``PreprocessConfig`` if the user confirms, or ``None`` if the user
        cancels at the confirmation step.

    In non-interactive environments, returns a default ``PreprocessConfig``
    immediately without any prompts.
    """
    if not _is_interactive():
        return default_config()

    console.print(
        Panel(
            "[bold]MLPrep[/bold] — Preprocessing Wizard\n"
            "Use arrow keys to select, Enter to confirm.",
            expand=False,
        )
    )

    numerical_strategy = ask_numerical_strategy()
    categorical_strategy = ask_categorical_strategy()

    # V2: encoding and scaling are fixed — shown as confirmed choices
    console.print("\n[bold cyan]Encoding:[/bold cyan]  ✓ One-Hot Encoding")
    console.print("[bold cyan]Scaling:[/bold cyan]   ✓ Standard Scaling\n")

    config = PreprocessConfig(
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy,
        encoding="onehot",
        scaling="standard",
    )

    confirmed = ask_confirmation(config)
    if not confirmed:
        return None

    return config