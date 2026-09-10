"""Tests for the preprocessing report."""

import json
from pathlib import Path

import pandas as pd

from mlprep.config import PreprocessConfig
from mlprep.models import datasetProfile
from mlprep.profiler import profile_dataset
from mlprep.reports import (
    PreprocessReport,
    build_report,
    display_report,
    save_report_json,
)


def _make_profile() -> datasetProfile:
    df = pd.DataFrame(
        {
            "age": [20.0, 25.0, 30.0],
            "salary": [50000.0, 60000.0, None],
            "city": ["Delhi", "Mumbai", "Delhi"],
        }
    )
    return profile_dataset(df)


def _make_report(tmp_path: Path) -> PreprocessReport:
    config = PreprocessConfig(
        numerical_strategy="median", categorical_strategy="most_frequent"
    )
    profile = _make_profile()
    return build_report(
        config=config,
        profile=profile,
        input_path=Path("data/sample.csv"),
        output_path=tmp_path / "processed.csv",
        pipeline_path=tmp_path / "preprocessor.joblib",
        report_path=tmp_path / "report.json",
    )


def test_report_has_correct_row_count(tmp_path):
    report = _make_report(tmp_path)
    assert report.rows == 3


def test_report_lists_numerical_columns(tmp_path):
    report = _make_report(tmp_path)
    assert "age" in report.numerical_columns
    assert "salary" in report.numerical_columns


def test_report_lists_categorical_columns(tmp_path):
    report = _make_report(tmp_path)
    assert "city" in report.categorical_columns


def test_report_transformations_describe_strategy(tmp_path):
    report = _make_report(tmp_path)
    age_tx = next(t for t in report.transformations if t.column == "age")
    assert "median" in age_tx.imputation
    assert "scaling" in age_tx.transformation


def test_report_categorical_transformation(tmp_path):
    report = _make_report(tmp_path)
    city_tx = next(t for t in report.transformations if t.column == "city")
    assert "most_frequent" in city_tx.imputation
    assert "encoding" in city_tx.transformation


def test_report_constant_strategy_described(tmp_path):
    config = PreprocessConfig(numerical_strategy="constant", numerical_fill_value=0.0)
    profile = _make_profile()
    report = build_report(
        config=config,
        profile=profile,
        input_path=Path("data/sample.csv"),
        output_path=tmp_path / "processed.csv",
        pipeline_path=tmp_path / "preprocessor.joblib",
        report_path=tmp_path / "report.json",
    )
    age_tx = next(t for t in report.transformations if t.column == "age")
    assert "constant" in age_tx.imputation
    assert "fill=0.0" in age_tx.imputation


def test_save_report_json(tmp_path):
    report = _make_report(tmp_path)
    report_path = tmp_path / "report.json"
    save_report_json(report, report_path)
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["rows"] == 3
    assert "age" in data["numerical_columns"]


def test_display_report_runs_without_error(tmp_path, capsys):
    """display_report should not raise any exceptions."""
    report = _make_report(tmp_path)
    display_report(report)  # Should not raise
