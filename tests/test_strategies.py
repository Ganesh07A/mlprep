"""Tests for preprocessing strategies — every numerical and categorical strategy."""

import pandas as pd
import pytest

from mlprep.config import PreprocessConfig
from mlprep.preprocess.pipeline import (
    create_preprocessor,
    preprocess_dataset,
)


def _make_df():
    return pd.DataFrame(
        {
            "age": [20.0, None, 30.0],
            "salary": [50000.0, 60000.0, None],
            "city": ["Delhi", None, "Mumbai"],
        }
    )


# ── Numerical strategies ──────────────────────────────────────────────────────


def test_numerical_median_strategy():
    df = _make_df()
    config = PreprocessConfig(numerical_strategy="median")
    processed_df, _ = preprocess_dataset(df, config)
    # After median imputation and standard scaling, no NaN values should remain
    assert not processed_df.isnull().any().any()


def test_numerical_mean_strategy():
    df = _make_df()
    config = PreprocessConfig(numerical_strategy="mean")
    processed_df, _ = preprocess_dataset(df, config)
    assert not processed_df.isnull().any().any()


def test_numerical_constant_strategy():
    df = _make_df()
    config = PreprocessConfig(numerical_strategy="constant", numerical_fill_value=0.0)
    processed_df, _ = preprocess_dataset(df, config)
    assert not processed_df.isnull().any().any()


def test_numerical_constant_custom_fill():
    """Constant fill with a specific value should influence scaling output."""
    df = pd.DataFrame({"score": [10.0, None, 30.0]})
    config = PreprocessConfig(numerical_strategy="constant", numerical_fill_value=-999.0)
    processed_df, _ = preprocess_dataset(df, config)
    assert not processed_df.isnull().any().any()
    # The imputed -999 will be a very negative scaled value
    assert processed_df["numerical__score"].min() < -1.0


# ── Categorical strategies ────────────────────────────────────────────────────


def test_categorical_most_frequent_strategy():
    df = _make_df()
    config = PreprocessConfig(categorical_strategy="most_frequent")
    processed_df, _ = preprocess_dataset(df, config)
    assert not processed_df.isnull().any().any()


def test_categorical_constant_strategy():
    df = _make_df()
    config = PreprocessConfig(categorical_strategy="constant")
    processed_df, _ = preprocess_dataset(df, config)
    assert not processed_df.isnull().any().any()


def test_categorical_constant_custom_fill():
    """Constant fill 'unknown' should produce an extra OHE column."""
    df = pd.DataFrame({"city": ["Delhi", None, "Mumbai"]})
    config = PreprocessConfig(
        categorical_strategy="constant", categorical_fill_value="unknown"
    )
    processed_df, _ = preprocess_dataset(df, config)
    # Should have a column for the 'unknown' category
    assert any("unknown" in col for col in processed_df.columns)


# ── Invalid strategy rejected by config ───────────────────────────────────────


def test_invalid_numerical_strategy_rejected():
    with pytest.raises(ValueError, match="Unsupported numerical strategy"):
        PreprocessConfig(numerical_strategy="bad_strategy")


def test_invalid_categorical_strategy_rejected():
    with pytest.raises(ValueError, match="Unsupported categorical strategy"):
        PreprocessConfig(categorical_strategy="bad_strategy")


# ── create_preprocessor uses config ──────────────────────────────────────────


def test_create_preprocessor_uses_config():
    config = PreprocessConfig(numerical_strategy="mean")
    preprocessor = create_preprocessor(["age"], ["city"], config)
    # Check the numerical imputer has the correct strategy
    num_pipeline = {name: est for name, est, _ in preprocessor.transformers}["numerical"]
    assert num_pipeline.named_steps["imputer"].strategy == "mean"
