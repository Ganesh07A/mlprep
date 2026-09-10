"""Tests for data quality checks."""

import pandas as pd

from mlprep.quality import (
    QualityWarning,
    run_quality_checks,
)


def _check_names(warnings: list[QualityWarning]) -> set[str]:
    return {w.check_name for w in warnings}


# ── High missingness ──────────────────────────────────────────────────────────


def test_high_missingness_detected():
    df = pd.DataFrame({"age": [None, None, None, 30.0, None]})  # 80% missing
    warnings = run_quality_checks(df)
    assert "high_missingness" in _check_names(warnings)


def test_low_missingness_not_flagged():
    df = pd.DataFrame({"age": [20.0, None, 30.0]})  # 33% — below 50% threshold
    warnings = run_quality_checks(df)
    assert "high_missingness" not in _check_names(warnings)


# ── Constant columns ──────────────────────────────────────────────────────────


def test_constant_column_detected():
    df = pd.DataFrame({"status": ["active", "active", "active"]})
    warnings = run_quality_checks(df)
    assert "constant_column" in _check_names(warnings)


def test_non_constant_column_not_flagged():
    df = pd.DataFrame({"status": ["active", "inactive", "active"]})
    warnings = run_quality_checks(df)
    assert "constant_column" not in _check_names(warnings)


# ── Duplicate rows ────────────────────────────────────────────────────────────


def test_duplicate_rows_detected():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    warnings = run_quality_checks(df)
    assert "duplicate_rows" in _check_names(warnings)


def test_no_duplicate_rows_not_flagged():
    df = pd.DataFrame({"a": [1, 2, 3]})
    warnings = run_quality_checks(df)
    assert "duplicate_rows" not in _check_names(warnings)


# ── Duplicate column names ────────────────────────────────────────────────────


def test_duplicate_column_names_detected():
    df = pd.DataFrame([[1, 2]], columns=["age", "age"])
    warnings = run_quality_checks(df)
    assert "duplicate_column_names" in _check_names(warnings)


# ── High cardinality ──────────────────────────────────────────────────────────


def test_high_cardinality_detected():
    # 60 unique string values
    df = pd.DataFrame({"city": [f"city_{i}" for i in range(60)]})
    warnings = run_quality_checks(df)
    assert "high_cardinality" in _check_names(warnings)


def test_low_cardinality_not_flagged():
    df = pd.DataFrame({"city": ["Delhi", "Mumbai", "Pune"] * 10})
    warnings = run_quality_checks(df)
    assert "high_cardinality" not in _check_names(warnings)


# ── Identifier-like columns ───────────────────────────────────────────────────


def test_identifier_column_by_name_detected():
    df = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5]})
    warnings = run_quality_checks(df)
    assert "identifier_column" in _check_names(warnings)


def test_identifier_column_by_uniqueness_detected():
    # Column with 100% unique values AND enough rows to trigger the check
    df = pd.DataFrame({"ref": list(range(100))})
    warnings = run_quality_checks(df)
    assert "identifier_column" in _check_names(warnings)


# ── Imbalanced categorical ────────────────────────────────────────────────────


def test_imbalanced_categorical_detected():
    # 98% "yes"
    df = pd.DataFrame({"label": ["yes"] * 98 + ["no"] * 2})
    warnings = run_quality_checks(df)
    assert "imbalanced_categorical" in _check_names(warnings)


def test_balanced_categorical_not_flagged():
    df = pd.DataFrame({"label": ["yes", "no"] * 50})
    warnings = run_quality_checks(df)
    assert "imbalanced_categorical" not in _check_names(warnings)


# ── Clean dataset ─────────────────────────────────────────────────────────────


def test_clean_dataset_returns_no_warnings():
    df = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0, 35.0, 40.0],
            "salary": [30000.0, 40000.0, 50000.0, 60000.0, 70000.0],
            "city": ["Delhi", "Mumbai", "Pune", "Chennai", "Kolkata"],
        }
    )
    warnings = run_quality_checks(df)
    # May or may not flag 'city' as high cardinality (5 values — well below 50)
    problematic = {w.check_name for w in warnings} - {"potential_datetime"}
    assert not problematic, f"Unexpected warnings on clean dataset: {problematic}"


# ── Warning structure ─────────────────────────────────────────────────────────


def test_quality_warning_has_required_fields():
    df = pd.DataFrame({"age": [None, None, None, 30.0, None]})
    warnings = run_quality_checks(df)
    w = next(w for w in warnings if w.check_name == "high_missingness")
    assert w.column == "age"
    assert w.message
    assert w.recommendation


def test_display_quality_warnings_no_error(capsys):
    from mlprep.quality import display_quality_warnings

    df = pd.DataFrame({"age": [None, None, None, 30.0, None]})
    warnings = run_quality_checks(df)
    display_quality_warnings(warnings)  # Should not raise
