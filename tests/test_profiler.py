import pandas as pd

from mlprep.profiler import profile_dataset


def test_profile_detects_missing_values():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )

    profile = profile_dataset(df)

    assert profile.missing_count["age"] == 1


def test_profile_detects_numerical_columns():
    df = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )

    profile = profile_dataset(df)

    assert profile.numerical_values == ["age"]


def test_profile_detects_categorical_columns():
    df = pd.DataFrame(
        {
            "age": [20, 25, 30],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )

    profile = profile_dataset(df)

    assert profile.categorical_values == ["city"]


def test_profile_calculates_missing_percentage():
    df = pd.DataFrame(
        {
            "age": [20, None, 30],
        }
    )

    profile = profile_dataset(df)

    assert profile.missing_percentage["age"] == (1 / 3) * 100
