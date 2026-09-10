"""Dataset loading utilities.

Separates loading logic from validation so both can be independently tested.
The CLI boundary handles user-facing error messages; this module raises
``ValueError`` only, keeping it free of UI concerns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded ``DataFrame``.

    Raises:
        ValueError: On empty file, parse errors, decode errors, or unreadable files.
    """
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError("The dataset is empty.")
    except pd.errors.ParserError as error:
        raise ValueError("Could not parse the CSV file. Is it valid CSV?") from error
    except UnicodeDecodeError as error:
        raise ValueError(
            "Could not decode the file. Try saving it as UTF-8."
        ) from error
    except PermissionError as error:
        raise ValueError(f"Permission denied reading: {file_path}") from error
    except OSError as error:
        raise ValueError(f"Could not read file: {error}") from error

    if df.empty:
        raise ValueError("The dataset is empty (no rows).")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    return df


def load_validated_dataset(file_path: Path) -> pd.DataFrame:
    """Validate then load a dataset.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Loaded ``DataFrame``.

    Raises:
        ValueError: On validation failure or load failure.
    """
    from mlprep.validator import validate_input

    validate_input(file_path)
    return load_dataset(file_path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Output path. Parent directories must exist.

    Raises:
        ValueError: On permission or disk errors.
    """
    try:
        df.to_csv(path, index=False)
    except PermissionError as error:
        raise ValueError(f"Permission denied writing to: {path}") from error
    except OSError as error:
        raise ValueError(f"Could not write file: {error}") from error
