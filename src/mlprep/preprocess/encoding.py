import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def create_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def encode_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include="object").columns

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    encoded_data = encoder.fit_transform(df[categorical_columns])

    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=df.index,
    )

    df = df.drop(columns=categorical_columns)

    return pd.concat([df, encoded_df], axis=1)
