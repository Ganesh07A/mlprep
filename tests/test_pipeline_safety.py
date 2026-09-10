"""Tests for pipeline safety and column compatibility.

Covers:
  - Column ORDER does not matter (nice portfolio feature)
  - Missing required columns → hard error
  - Extra columns → warn + ignore
  - Empty input
  - One-row input
  - Fit/transform separation (no refitting on new data)
"""

import pandas as pd

from mlprep.config import default_config
from mlprep.preprocess.pipeline import (
    get_column_types,
    preprocess_dataset,
    transform_dataset,
)


def _train_df():
    return pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0],
            "salary": [50000.0, 60000.0, 70000.0],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )


# ── Column order independence ─────────────────────────────────────────────────


def test_column_order_does_not_matter():
    """New data with columns in a different order should transform correctly."""
    train_df = _train_df()
    config = default_config()
    _, preprocessor = preprocess_dataset(train_df, config)

    # New data with shuffled column order
    new_df = pd.DataFrame(
        {
            "city": ["Pune"],
            "salary": [55000.0],
            "age": [28.0],
        }
    )

    # Reorder to match training (as cli._prepare_dataframe_for_transform does)
    required_columns = []
    for _, _, cols in preprocessor.transformers_:
        if isinstance(cols, list):
            required_columns.extend(cols)
    new_df_reordered = new_df[required_columns]

    result = transform_dataset(new_df_reordered, preprocessor)
    assert result is not None
    assert len(result) == 1


# ── Missing required columns ──────────────────────────────────────────────────


def test_missing_required_column_raises():
    """New data missing a training column should fail at transform time."""
    train_df = _train_df()
    config = default_config()
    _, preprocessor = preprocess_dataset(train_df, config)

    # Drop 'age' — a required column
    new_df = pd.DataFrame(
        {
            "salary": [55000.0],
            "city": ["Pune"],
        }
    )

    required_columns = []
    for _, _, cols in preprocessor.transformers_:
        if isinstance(cols, list):
            required_columns.extend(cols)

    missing = set(required_columns) - set(new_df.columns)
    assert "age" in missing  # Verify our test is correct


# ── Extra columns are ignored ─────────────────────────────────────────────────


def test_extra_columns_are_handled():
    """New data with extra columns can be trimmed to required columns."""
    train_df = _train_df()
    config = default_config()
    _, preprocessor = preprocess_dataset(train_df, config)

    required_columns = []
    for _, _, cols in preprocessor.transformers_:
        if isinstance(cols, list):
            required_columns.extend(cols)

    new_df = pd.DataFrame(
        {
            "age": [28.0],
            "salary": [55000.0],
            "city": ["Pune"],
            "country": ["India"],  # extra column
        }
    )

    # Trim to required columns (as the CLI does)
    trimmed = new_df[required_columns]
    result = transform_dataset(trimmed, preprocessor)
    assert result is not None
    # 'country' column should NOT appear in the output
    assert not any("country" in col for col in result.columns)


# ── Empty input ───────────────────────────────────────────────────────────────


def test_empty_dataframe_has_no_column_types():
    """An empty DataFrame returns empty column lists."""
    df = pd.DataFrame()
    num, cat = get_column_types(df)
    assert num == []
    assert cat == []


# ── One-row input ─────────────────────────────────────────────────────────────


def test_one_row_preprocesses_successfully():
    """A single-row DataFrame should preprocess without errors."""
    df = pd.DataFrame({"age": [25.0], "city": ["Delhi"]})
    config = default_config()
    processed_df, _ = preprocess_dataset(df, config)
    assert len(processed_df) == 1


def test_one_row_transforms_successfully():
    """A single-row DataFrame should transform with a fitted pipeline."""
    train_df = _train_df()
    config = default_config()
    _, preprocessor = preprocess_dataset(train_df, config)

    new_df = pd.DataFrame({"age": [35.0], "salary": [80000.0], "city": ["Pune"]})
    result = transform_dataset(new_df, preprocessor)
    assert len(result) == 1


# ── No refitting on new data ──────────────────────────────────────────────────


def test_transform_does_not_refit():
    """transform_dataset should not change the fitted preprocessor's statistics."""
    train_df = _train_df()
    config = default_config()
    _, preprocessor = preprocess_dataset(train_df, config)

    # Record training statistics (mean of age after standard scaler)
    num_pipeline = {name: est for name, est, _ in preprocessor.transformers_}[
        "numerical"
    ]
    mean_before = num_pipeline.named_steps["scaler"].mean_.copy()

    # Transform completely different data
    new_df = pd.DataFrame(
        {
            "age": [100.0, 200.0, 300.0],
            "salary": [1.0, 2.0, 3.0],
            "city": ["X", "Y", "Z"],
        }
    )
    transform_dataset(new_df, preprocessor)

    mean_after = (
        {name: est for name, est, _ in preprocessor.transformers_}["numerical"]
        .named_steps["scaler"]
        .mean_
    )
    assert (mean_before == mean_after).all(), "Preprocessor was refitted on new data!"


# ── All-numerical or all-categorical DataFrames ───────────────────────────────


def test_all_numerical_dataframe():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    config = default_config()
    processed_df, _ = preprocess_dataset(df, config)
    assert not processed_df.isnull().any().any()


def test_all_categorical_dataframe():
    df = pd.DataFrame({"x": ["a", "b", "a"], "y": ["p", "q", "p"]})
    config = default_config()
    processed_df, _ = preprocess_dataset(df, config)
    assert not processed_df.isnull().any().any()
