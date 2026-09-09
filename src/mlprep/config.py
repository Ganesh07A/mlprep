"""Configuration model for MLPrep preprocessing.

Every interface (interactive wizard, expert CLI flags, YAML file) produces a
``PreprocessConfig`` object.  The pipeline engine consumes only this object,
so the same preprocessing behaviour is guaranteed regardless of how the user
invoked MLPrep.

Security note
─────────────
``config.py`` itself is safe; it only reads YAML text.  The downstream
``preprocessor.joblib`` file is produced by Python's ``joblib`` / ``pickle``
serialisation.  **Only load ``.joblib`` files that you generated yourself with
MLPrep.** Loading an untrusted ``.joblib`` file can execute arbitrary code,
just like ``pickle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# ── Strategy constants ────────────────────────────────────────────────────────

NUMERICAL_STRATEGIES: tuple[str, ...] = ("median", "mean", "constant")
CATEGORICAL_STRATEGIES: tuple[str, ...] = ("most_frequent", "constant")

# Fixed choices for V2 — architecture supports extension later
ENCODING_STRATEGIES: tuple[str, ...] = ("onehot",)
SCALING_STRATEGIES: tuple[str, ...] = ("standard",)

# Default fill values when strategy == "constant"
NUMERICAL_CONSTANT_FILL: float = 0.0
CATEGORICAL_CONSTANT_FILL: str = "missing"


# ── Config model ──────────────────────────────────────────────────────────────


@dataclass
class PreprocessConfig:
    """Preprocessing configuration.

    All fields have sensible defaults so callers only override what they need.
    """

    numerical_strategy: str = "median"
    categorical_strategy: str = "most_frequent"
    encoding: str = "onehot"
    scaling: str = "standard"
    numerical_fill_value: float | None = None
    categorical_fill_value: str | None = None

    def __post_init__(self) -> None:
        validate_config(self)

    @property
    def resolved_numerical_fill(self) -> float:
        """Return the fill value for constant numerical imputation."""
        return (
            self.numerical_fill_value
            if self.numerical_fill_value is not None
            else NUMERICAL_CONSTANT_FILL
        )

    @property
    def resolved_categorical_fill(self) -> str:
        """Return the fill value for constant categorical imputation."""
        return (
            self.categorical_fill_value
            if self.categorical_fill_value is not None
            else CATEGORICAL_CONSTANT_FILL
        )


# ── Validation ────────────────────────────────────────────────────────────────


def validate_config(config: PreprocessConfig) -> None:
    """Raise ``ValueError`` if any strategy in *config* is unsupported."""
    if config.numerical_strategy not in NUMERICAL_STRATEGIES:
        raise ValueError(
            f"Unsupported numerical strategy '{config.numerical_strategy}'. "
            f"Choose from: {', '.join(NUMERICAL_STRATEGIES)}"
        )
    if config.categorical_strategy not in CATEGORICAL_STRATEGIES:
        raise ValueError(
            f"Unsupported categorical strategy '{config.categorical_strategy}'. "
            f"Choose from: {', '.join(CATEGORICAL_STRATEGIES)}"
        )
    if config.encoding not in ENCODING_STRATEGIES:
        raise ValueError(
            f"Unsupported encoding '{config.encoding}'. "
            f"Choose from: {', '.join(ENCODING_STRATEGIES)}"
        )
    if config.scaling not in SCALING_STRATEGIES:
        raise ValueError(
            f"Unsupported scaling '{config.scaling}'. "
            f"Choose from: {', '.join(SCALING_STRATEGIES)}"
        )


# ── YAML loader ───────────────────────────────────────────────────────────────


def load_config(path: Path) -> PreprocessConfig:
    """Load a YAML configuration file and return a validated ``PreprocessConfig``.

    Missing YAML fields fall back to ``PreprocessConfig`` defaults so a minimal
    config file is perfectly valid.

    Example YAML::

        missing_values:
          numerical: median
          categorical: most_frequent

        encoding:
          method: onehot

        scaling:
          method: standard

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        A validated ``PreprocessConfig``.

    Raises:
        ValueError: If the file cannot be read, is invalid YAML, or contains
                    unsupported strategy names.
    """
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {exc}") from exc

    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise TypeError("Config file must be a YAML mapping at the top level.")

    missing = raw.get("missing_values", {}) or {}
    encoding_section = raw.get("encoding", {}) or {}
    scaling_section = raw.get("scaling", {}) or {}

    return PreprocessConfig(
        numerical_strategy=missing.get("numerical", "median"),
        categorical_strategy=missing.get("categorical", "most_frequent"),
        encoding=encoding_section.get("method", "onehot"),
        scaling=scaling_section.get("method", "standard"),
    )


# ── Default config factory ────────────────────────────────────────────────────


def default_config() -> PreprocessConfig:
    """Return a ``PreprocessConfig`` with all defaults applied."""
    return PreprocessConfig()
