from pathlib import Path

import joblib
import typer
from rich.console import Console

from mlprep.display import display_profile
from mlprep.io import load_validated_dataset
from mlprep.preprocess.pipeline import create_preprocessor, create_processed_dataframe
from mlprep.validator import validate_input

app = typer.Typer()
console = Console()


@app.command()
def inspect(file_path: Path):
    try:
        validate_input(file_path)
        df = load_validated_dataset(file_path)
        display_profile(df)

    except ValueError as error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(code=1)


@app.command()
def preprocess(
    file_path: Path,
    output: Path = Path("./data/processed.csv"),
) -> None:

    try:
        validate_input(file_path)
        df = load_validated_dataset(file_path)

    except ValueError as error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(code=1)

    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "str"]).columns.tolist()

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    processed_data = preprocessor.fit_transform(df)

    processed_df = create_processed_dataframe(
        processed_data,
        preprocessor,
        df.index,
    )

    # Ask before overwriting an existing processed file
    if output.exists():
        overwrite = typer.confirm(f"{output} already exists. Overwrite?")

        if not overwrite:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    # Create output directory if it doesn't exist
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    processed_df.to_csv(output, index=False)

    # Save fitted preprocessing pipeline
    pipeline_output = output.parent / "preprocessor.joblib"
    joblib.dump(preprocessor, pipeline_output)

    console.print(f"[green]Processed dataset saved to: {output}[/green]")
    console.print(f"[green]Preprocessing pipeline saved to: {pipeline_output}[/green]")


@app.command()
def transform(
    file_path: Path,
    pipeline: Path,
    output: Path = Path("./data/transformed.csv"),
) -> None:

    try:
        load_validated_dataset(file_path)

        if not pipeline.exists():
            raise ValueError("Pipeline file does not exist.")

        df = load_validated_dataset(file_path)
        preprocessor = joblib.load(pipeline)

    except ValueError as error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(code=1)

    processed_data = preprocessor.transform(df)

    processed_df = create_processed_dataframe(
        processed_data,
        preprocessor,
        df.index,
    )

    # Ask before overwriting an existing transformed file
    if output.exists():
        overwrite = typer.confirm(f"{output} already exists. Overwrite?")

        if not overwrite:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit()

    # Create output directory if it doesn't exist
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save transformed dataset
    processed_df.to_csv(output, index=False)

    console.print(f"[green]Transformed dataset saved to: {output}[/green]")


if __name__ == "__main__":
    app()
