import pandas as pd
from sklearn.impute import SimpleImputer


def get_missing_columns(df: pd.DataFrame) -> list[str]:
    return df.columns[df.isnull().any()].tolist()


# func for filling the null values
def create_numerical_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median")


# filling numerical column null values with median by imputer
def impute_numerical_data(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns = df.select_dtypes(include="number").columns

    imputer = SimpleImputer(strategy="median")

    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    return df


def impute_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include="object").columns

    imputer = SimpleImputer(strategy="most_frequent")

    df[categorical_columns] = imputer.fit_transform(df[categorical_columns])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = impute_categorical_data(df)
    df = impute_numerical_data(df)
    return df
