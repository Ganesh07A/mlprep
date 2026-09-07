from pathlib import Path
import pandas as pd


def load_dataset(file_path:Path) -> pd.DataFrame:
    try:
        df =  pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("The datset is Empty!")
    except len(pd.columns == 0) as error:
        raise ValueError("The datset has no columns!")
    except pd.errors.ParserError as error:
        raise ValueError("Could not load the dataset.") from error
    except UnicodeDecodeError as error:
        raise ValueError("Could not decode the csv file.") from error

    # if df.empty: raise ValueError("The dataset is empty!")
    
    return df