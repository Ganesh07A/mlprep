"""Preprocessing pipeline engine.

The single entry-point is ``preprocess_dataset(df, config)``.  All interfaces
(interactive wizard, expert CLI flags, YAML config) converge on a
``PreprocessConfig`` object before calling into this module.

Architecture
────────────

                Raw Data
                   │
           Column classification
                   │
       ┌───────────┴───────────┐
       │                       │
  Numerical               Categorical
       │                       │
   Imputer                  Imputer
       │                       │
    Scaler                  Encoder
       │                       │
       └───────────┬───────────┘
                   │
            Processed Data

Rule: fit() on training data, transform() on new data — never refit new data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlprep.config import PreprocessConfig, default_config

# ── Column classification ─────────────────────────────────────────────────────


def get_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numerical_columns, categorical_columns) for *df*."""
    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = (
        df.select_dtypes(include=["str", "category"]).columns.tolist()
    )
    return numerical_columns, categorical_columns


# ── Pipeline factory ──────────────────────────────────────────────────────────


def create_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
    config: PreprocessConfig | None = None,
) -> ColumnTransformer:
    """Build a fitted-ready ``ColumnTransformer`` from column lists and config.

    Args:
        numerical_columns: List of numerical column names.
        categorical_columns: List of categorical column names.
        config: ``PreprocessConfig`` that drives strategy selection. Defaults
                to ``default_config()`` (all defaults) when omitted.

    Returns:
        An unfitted ``ColumnTransformer`` ready for ``fit_transform`` or ``fit``.
    """
    if config is None:
        config = default_config()

    # Resolve imputer kwargs — constant strategy needs fill_value
    num_imputer_kwargs: dict = {"strategy": config.numerical_strategy}
    if config.numerical_strategy == "constant":
        num_imputer_kwargs["fill_value"] = config.resolved_numerical_fill

    cat_imputer_kwargs: dict = {"strategy": config.categorical_strategy}
    if config.categorical_strategy == "constant":
        cat_imputer_kwargs["fill_value"] = config.resolved_categorical_fill

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(**num_imputer_kwargs)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(**cat_imputer_kwargs)),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numerical_columns:
        transformers.append(("numerical", numerical_pipeline, numerical_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers, remainder="drop")


# ── Output dataframe builder ──────────────────────────────────────────────────


def create_processed_dataframe(
    processed_data,
    preprocessor: ColumnTransformer,
    index,
) -> pd.DataFrame:
    """Wrap the raw numpy output of a ``ColumnTransformer`` into a DataFrame."""
    columns = preprocessor.get_feature_names_out()
    return pd.DataFrame(processed_data, columns=columns, index=index)


# ── High-level API ────────────────────────────────────────────────────────────


def preprocess_dataset(
    df: pd.DataFrame,
    config: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Fit a preprocessing pipeline on *df* and return the processed data.

    This is the **training** step: ``fit_transform`` is called once here.
    Use ``transform_dataset`` for new data.

    Args:
        df: Raw training ``DataFrame``.
        config: Preprocessing configuration. Uses ``default_config()`` when
                omitted.

    Returns:
        A tuple of (processed_df, fitted_preprocessor).
    """
    if config is None:
        config = default_config()

    numerical_columns, categorical_columns = get_column_types(df)

    preprocessor = create_preprocessor(numerical_columns, categorical_columns, config)
    processed_data = preprocessor.fit_transform(df)
    processed_df = create_processed_dataframe(processed_data, preprocessor, df.index)

    return processed_df, preprocessor


def transform_dataset(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> pd.DataFrame:
    """Apply a *fitted* preprocessor to new data without refitting.

    This is the **inference** step: only ``transform`` is called.

    Args:
        df: New ``DataFrame`` to transform.
        preprocessor: A ``ColumnTransformer`` that was previously fitted with
                      ``preprocess_dataset``.

    Returns:
        Processed ``DataFrame`` with the same column schema as the training output.
    """
    processed_data = preprocessor.transform(df)
    return create_processed_dataframe(processed_data, preprocessor, df.index)