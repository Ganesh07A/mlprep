"""Tests for PreprocessConfig and the YAML config loader."""

from pathlib import Path

import pytest
import yaml

from mlprep.config import (
    CATEGORICAL_STRATEGIES,
    NUMERICAL_STRATEGIES,
    PreprocessConfig,
    default_config,
    load_config,
)

# ── Default config ────────────────────────────────────────────────────────────


def test_default_config_is_valid():
    config = default_config()
    assert config.numerical_strategy == "median"
    assert config.categorical_strategy == "most_frequent"
    assert config.encoding == "onehot"
    assert config.scaling == "standard"


# ── Strategy validation ───────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", NUMERICAL_STRATEGIES)
def test_all_numerical_strategies_are_valid(strategy):
    config = PreprocessConfig(numerical_strategy=strategy)
    assert config.numerical_strategy == strategy


@pytest.mark.parametrize("strategy", CATEGORICAL_STRATEGIES)
def test_all_categorical_strategies_are_valid(strategy):
    config = PreprocessConfig(categorical_strategy=strategy)
    assert config.categorical_strategy == strategy


def test_invalid_numerical_strategy_raises():
    with pytest.raises(ValueError, match="Unsupported numerical strategy"):
        PreprocessConfig(numerical_strategy="interpolate")


def test_invalid_categorical_strategy_raises():
    with pytest.raises(ValueError, match="Unsupported categorical strategy"):
        PreprocessConfig(categorical_strategy="random")


def test_invalid_encoding_raises():
    with pytest.raises(ValueError, match="Unsupported encoding"):
        PreprocessConfig(encoding="label")


def test_invalid_scaling_raises():
    with pytest.raises(ValueError, match="Unsupported scaling"):
        PreprocessConfig(scaling="minmax")


# ── Constant fill values ──────────────────────────────────────────────────────


def test_constant_numerical_fill_default():
    config = PreprocessConfig(numerical_strategy="constant")
    assert config.resolved_numerical_fill == 0.0


def test_constant_numerical_fill_custom():
    config = PreprocessConfig(
        numerical_strategy="constant", numerical_fill_value=-1.0
    )
    assert config.resolved_numerical_fill == -1.0


def test_constant_categorical_fill_default():
    config = PreprocessConfig(categorical_strategy="constant")
    assert config.resolved_categorical_fill == "missing"


def test_constant_categorical_fill_custom():
    config = PreprocessConfig(
        categorical_strategy="constant", categorical_fill_value="unknown"
    )
    assert config.resolved_categorical_fill == "unknown"


# ── YAML loading ──────────────────────────────────────────────────────────────


def test_load_full_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump(
            {
                "missing_values": {"numerical": "mean", "categorical": "constant"},
                "encoding": {"method": "onehot"},
                "scaling": {"method": "standard"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(cfg_file)
    assert config.numerical_strategy == "mean"
    assert config.categorical_strategy == "constant"
    assert config.encoding == "onehot"
    assert config.scaling == "standard"


def test_load_minimal_config_uses_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("{}", encoding="utf-8")
    config = load_config(cfg_file)
    assert config.numerical_strategy == "median"
    assert config.categorical_strategy == "most_frequent"


def test_load_empty_config_uses_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("", encoding="utf-8")
    config = load_config(cfg_file)
    assert config.numerical_strategy == "median"


def test_load_config_invalid_strategy_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"missing_values": {"numerical": "INVALID"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported numerical strategy"):
        load_config(cfg_file)


def test_load_config_missing_file_raises():
    with pytest.raises(ValueError, match="Config file not found"):
        load_config(Path("/nonexistent/config.yaml"))


def test_load_config_bad_yaml_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(": invalid: yaml: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(cfg_file)


def test_load_config_non_mapping_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_config(cfg_file)
