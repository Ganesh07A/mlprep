import pandas as pd

from mlprep.models import datasetProfile


def profile_dataset(df: pd.DataFrame) -> datasetProfile:

    # classification of column values
    categorical_values = df.select_dtypes(include=["object", "str"]).columns.tolist()
    numerical_values = df.select_dtypes(include="number").columns.tolist()

    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_count = missing_count[missing_count > 0]
    missing_percentage = missing_percentage[missing_percentage > 0]

    return datasetProfile(
        rows=len(df),
        columns=len(df.columns),
        missing_count=missing_count,
        missing_percentage=missing_percentage,
        data_types=df.dtypes,
        sample_data=df.head(),
        categorical_values=categorical_values,
        numerical_values=numerical_values,
    )
