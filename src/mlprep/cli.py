from pathlib import Path
from mlprep.display import display_profile
from mlprep.io import load_dataset
from mlprep.validator import validate_input
import typer
import pandas as pd
from rich.console import Console
app = typer.Typer()
console=Console()

@app.command()
def inspect(file_path:Path):

    try:
        validate_input(file_path)
    except ValueError as error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(code=1)

    try:
        df = load_dataset(file_path)
        display_profile(df)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")


    


    # console.print(f"\n[bold]DATASET SUMMARY[/bold]")
    # console.print(f"Rows: {profile.rows}")
    # console.print(f"Columns: {profile.columns}")
    # console.print(f"Numerical columns: {len(profile.numerical_values)}")
    # console.print(f"Categorical columns: {len(profile.categorical_values)}")

if __name__ == "__main__":
    app()