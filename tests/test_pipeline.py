import pandas as pd

from mlprep.preprocess.pipeline import create_preprocessor, create_processed_dataframe


def test_preprocessor_handles_missing_numeric_values():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )

    preprocessor = create_preprocessor(
        numerical_columns=["age"],
        categorical_columns=["city"],
    )

    processed = preprocessor.fit_transform(df)

    assert not pd.isna(processed).any()


def test_preprocessor_encodes_categorical_values():

    df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )

    preprocessor = create_preprocessor(
        numerical_columns=["age"],
        categorical_columns=["city"],
    )

    preprocessor.fit(df)

    processed = preprocessor.transform(df)

    assert processed.shape[1] == 3


def test_preprocessor_handles_unknown_category():
    train_df = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )

    new_df = pd.DataFrame(
        {
            "age": [35],
            "city": ["Pune"],
        }
    )

    preprocessor = create_preprocessor(
        numerical_columns=["age"],
        categorical_columns=["city"],
    )

    preprocessor.fit(train_df)

    processed = preprocessor.transform(new_df)

    assert processed.shape == (1, 3)


def test_preprocessor_scales_numeric_values():
    df = pd.DataFrame(
        {
            "age": [20, 25, 30],
        }
    )

    preprocessor = create_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    processed = preprocessor.fit_transform(df)

    assert processed[0][0] < 0
    assert processed[1][0] == 0
    assert processed[2][0] > 0


def test_create_processed_dataframe():
    df = pd.DataFrame(
        {
            "age": [20, 25],
        }
    )

    preprocessor = create_preprocessor(
        numerical_columns=["age"],
        categorical_columns=[],
    )

    processed_data = preprocessor.fit_transform(df)

    processed_df = create_processed_dataframe(
        processed_data,
        preprocessor,
        df.index,
    )

    assert isinstance(processed_df, pd.DataFrame)
    assert list(processed_df.columns) == ["numerical__age"]
    assert len(processed_df) == 2
