from dataclasses import dataclass

import pandas as pd


@dataclass
class datasetProfile:
    rows: int
    columns: int
    missing_count: pd.Series
    missing_percentage: pd.Series
    data_types: pd.Series
    sample_data: pd.DataFrame
    categorical_values: list[str]
    numerical_values: list[str]
