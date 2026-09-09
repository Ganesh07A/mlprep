from typer.testing import CliRunner

from mlprep.cli import app

runner = CliRunner()


def test_inspect_command():
    result = runner.invoke(app, ["inspect", "data/sample.csv"])

    assert result.exit_code == 0
    assert "MISSING VALUES" in result.stdout


def test_inspect_invalid_file():
    result = runner.invoke(app, ["inspect", "data/not_found.csv"])

    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_preprocess_command(tmp_path):
    output = tmp_path / "processed.csv"

    result = runner.invoke(
        app,
        [
            "preprocess",
            "data/sample.csv",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Processed dataset saved" in result.stdout


def test_transform_command(tmp_path):
    output = tmp_path / "transformed.csv"

    result = runner.invoke(
        app,
        [
            "transform",
            "data/sample.csv",
            "data/preprocessor.joblib",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Transformed dataset saved" in result.stdout


def test_preprocess_saves_pipeline(tmp_path):
    output = tmp_path / "processed.csv"

    result = runner.invoke(
        app,
        [
            "preprocess",
            "data/sample.csv",
            "--output",
            str(output),
        ],
    )

    pipeline = output.parent / "preprocessor.joblib"

    assert result.exit_code == 0
    assert output.exists()
    assert pipeline.exists()


def test_saved_pipeline_can_transform_data(tmp_path):
    output = tmp_path / "processed.csv"

    result = runner.invoke(
        app,
        [
            "preprocess",
            "data/sample.csv",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0

    pipeline = output.parent / "preprocessor.joblib"

    new_data = tmp_path / "new_data.csv"
    new_data.write_text("name,age,city,salary\nEve,30,Pune,55000\n")

    result = runner.invoke(
        app,
        [
            "transform",
            str(new_data),
            str(pipeline),
            "--output",
            str(tmp_path / "transformed.csv"),
        ],
    )

    assert result.exit_code == 0
