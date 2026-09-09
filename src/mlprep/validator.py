from pathlib import Path

import typer
from rich.console import Console

console = Console()
SUPPORTED_EXTENSIONS = {".csv"}


def validate_input(file_path: Path) -> None:
    if not file_path.exists():
        console.print("[red]Error: File does not exist.[/red]")
        raise typer.Exit(code=1)

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print("[red]Error: Only csv files are accepted.[/red]")
        raise typer.Exit(code=1)
