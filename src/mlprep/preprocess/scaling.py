import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_numerical_data(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = df.select_dtypes(include="number").columns

    scaler = StandardScaler()

    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df
