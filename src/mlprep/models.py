from pandas.core.arrays import categorical
from dataclasses import dataclass
import pandas as pd

@dataclass
class datasetProfile:
    missing_values: pd.Series
    data_types: pd.Series
    sample_data:pd.DataFrame
    categorical_values: list[str]
    numerical_values: list[str]