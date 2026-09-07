from pathlib import Path
import pandas as pd


def load_dataset(file_path:Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except pd.errors.ParserError as error:
        raise ValueError("Could not load the dataset.") from error
    except UnicodeDecodeError as error:
        raise ValueError("Could not decode the csv file.") from error