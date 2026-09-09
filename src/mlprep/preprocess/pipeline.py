import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def create_processed_dataframe(
    processed_data,
    preprocessor,
    index,
) -> pd.DataFrame:
    columns = preprocessor.get_feature_names_out()

    return pd.DataFrame(
        processed_data,
        columns=columns,
        index=index,
    )
