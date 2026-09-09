"""CLI integration tests.

Covers:
  inspect    — success, invalid file
  preprocess — default (non-interactive), expert flags, config file,
               config + flags rejection, invalid strategy
  transform  — success, invalid pipeline, missing required column
  TTY / mode detection tests via CliRunner
"""


import yaml
from typer.testing import CliRunner

from mlprep.cli import app

runner = CliRunner()


# ── inspect ───────────────────────────────────────────────────────────────────


def test_inspect_command():
    result = runner.invoke(app, ["inspect", "data/sample.csv"])
    assert result.exit_code == 0
    assert "MISSING VALUES" in result.stdout


def test_inspect_invalid_file():
    result = runner.invoke(app, ["inspect", "data/not_found.csv"])
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Error" in (result.stderr or "")


# ── preprocess — default mode (non-interactive, uses defaults) ────────────────


def test_preprocess_command(tmp_path):
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        ["preprocess", "data/sample.csv", "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert "Processed dataset saved" in result.stdout


def test_preprocess_saves_pipeline(tmp_path):
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        ["preprocess", "data/sample.csv", "--output", str(output)],
    )
    pipeline = output.parent / "preprocessor.joblib"
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert pipeline.exists()


def test_preprocess_saves_report_json(tmp_path):
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        ["preprocess", "data/sample.csv", "--output", str(output)],
    )
    report = output.parent / "report.json"
    assert result.exit_code == 0, result.output
    assert report.exists()


# ── preprocess — expert flags ─────────────────────────────────────────────────


def test_preprocess_expert_numerical_strategy(tmp_path):
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        [
            "preprocess", "data/sample.csv",
            "--output", str(output),
            "--strategy", "mean",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mean" in result.stdout


def test_preprocess_expert_invalid_strategy(tmp_path):
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        [
            "preprocess", "data/sample.csv",
            "--output", str(output),
            "--strategy", "bad_strategy",
        ],
    )
    assert result.exit_code == 1


def test_preprocess_expert_cat_strategy(tmp_path):
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        [
            "preprocess", "data/sample.csv",
            "--output", str(output),
            "--cat-strategy", "constant",
        ],
    )
    assert result.exit_code == 0, result.output


# ── preprocess — config mode ──────────────────────────────────────────────────


def test_preprocess_config_mode(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump({
            "missing_values": {"numerical": "mean", "categorical": "constant"},
            "encoding": {"method": "onehot"},
            "scaling": {"method": "standard"},
        }),
        encoding="utf-8",
    )
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        [
            "preprocess", "data/sample.csv",
            "--output", str(output),
            "--config", str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output


def test_preprocess_config_invalid_strategy_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.dump({"missing_values": {"numerical": "bad"}}),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["preprocess", "data/sample.csv", "--config", str(cfg)],
    )
    assert result.exit_code == 1


# ── preprocess — config + flags rejection ────────────────────────────────────


def test_preprocess_config_plus_strategy_rejected(tmp_path):
    """--config combined with --strategy must be rejected with exit code 1."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "preprocess", "data/sample.csv",
            "--config", str(cfg),
            "--strategy", "mean",
        ],
    )
    assert result.exit_code == 1
    assert "Cannot combine" in result.stdout or "Cannot combine" in (result.stderr or "")


def test_preprocess_config_plus_cat_strategy_rejected(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "preprocess", "data/sample.csv",
            "--config", str(cfg),
            "--cat-strategy", "constant",
        ],
    )
    assert result.exit_code == 1


# ── transform ─────────────────────────────────────────────────────────────────


def test_transform_command(tmp_path):
    # First preprocess to get a pipeline
    output = tmp_path / "processed.csv"
    runner.invoke(
        app,
        ["preprocess", "data/sample.csv", "--output", str(output)],
    )
    pipeline_path = output.parent / "preprocessor.joblib"

    result = runner.invoke(
        app,
        [
            "transform",
            "data/sample.csv",
            str(pipeline_path),
            "--output", str(tmp_path / "transformed.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Transformed dataset saved" in result.stdout


def test_transform_invalid_pipeline_extension(tmp_path):
    fake_pipeline = tmp_path / "model.pkl"
    fake_pipeline.write_bytes(b"")
    result = runner.invoke(
        app,
        ["transform", "data/sample.csv", str(fake_pipeline)],
    )
    assert result.exit_code == 1


def test_transform_missing_pipeline_file(tmp_path):
    result = runner.invoke(
        app,
        ["transform", "data/sample.csv", str(tmp_path / "nonexistent.joblib")],
    )
    assert result.exit_code == 1


def test_saved_pipeline_can_transform_data(tmp_path):
    output = tmp_path / "processed.csv"
    result = runner.invoke(
        app,
        ["preprocess", "data/sample.csv", "--output", str(output)],
    )
    assert result.exit_code == 0, result.output

    pipeline = output.parent / "preprocessor.joblib"
    new_data = tmp_path / "new_data.csv"
    new_data.write_text("name,age,city,salary\nEve,30,Pune,55000\n")

    result = runner.invoke(
        app,
        [
            "transform",
            str(new_data),
            str(pipeline),
            "--output", str(tmp_path / "transformed.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
