# Changelog

All notable changes to MLPrep are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-09-10

### Added (V3 — Production / Portfolio Ready)

- **Data quality checks** (`quality.py`): 8 checks for common ML anti-patterns
  (high missingness, constant columns, duplicate rows/column names, high
  cardinality, identifier-like columns, potential datetimes, imbalanced
  categoricals). Warnings are recommendations only — data is never modified.
- **`--format json`** on `inspect`: machine-readable output for automation and CI.
- **`--verbose` / `-v`** flag on all commands: exposes internal log messages for
  debugging without cluttering normal terminal output.
- **`--no-warnings`** flag: suppress data quality warnings for automated pipelines.
- **Step messages** during `preprocess`: `✓ Dataset loaded`, `✓ Config ready`,
  `✓ Preprocessing complete`, and final output paths.
- **Logging module** (`logging_config.py`): `setup_logging(verbose)` configures
  Python logging with INFO/WARNING/ERROR/DEBUG levels. Intentionally separate
  from Rich terminal output.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): runs Ruff lint, Ruff
  format check, and pytest on every push/PR to main.
- **Full README**: problem statement, architecture diagram, feature table,
  installation, quick start, all 3 usage modes with examples, quality warnings
  table, outputs table, security note, limitations, and roadmap.
- **`save_dataframe()`** helper in `io.py`: handles permission and disk errors.
- **Quality tests** (`test_quality.py`): 18 tests covering all 8 checks.

### Changed

- `validator.py`: raises `ValueError` instead of calling `typer.Exit` directly.
  Fully decoupled from UI concerns; independently testable.
- `io.py`: rewrote `load_dataset()` — removed broken `except len(...)` clause,
  added handling for `PermissionError`, `OSError`, empty DataFrame, and
  zero-column DataFrame.
- `display.py`: split into `display_profile_rich()` and `display_profile_json()`.
  Added `display_profile(df, fmt)` dispatcher.
- `cli.py`: full rewrite with verbose logging, quality checks, step messages,
  richer help text with examples, `--no-warnings`, `--format json`, and
  write-error handling via `save_dataframe()`.
- `pyproject.toml`: bumped version to `1.0.0`, added description, license,
  authors, keywords, classifiers.

---

## [0.2.0] — 2026-09-09

### Added (V2 — Core ML Tool)

- **`PreprocessConfig`** dataclass (`config.py`) as the central config object.
  All three interfaces (interactive wizard, expert CLI flags, YAML) converge
  on `PreprocessConfig` before hitting the pipeline engine.
- **Strategy validation**: `NUMERICAL_STRATEGIES`, `CATEGORICAL_STRATEGIES`
  constants. Invalid strategies raise `ValueError` immediately at config
  construction time.
- **YAML config loader** (`load_config`): validates strategy names, applies
  defaults for missing fields, rejects non-YAML, invalid YAML, non-mapping YAML.
- **Interactive preprocessing wizard** (`ui.py`): `questionary`-based prompts
  with TTY detection (real terminal → wizard; CI/tests → defaults).
- **Expert CLI mode**: `--strategy`, `--cat-strategy` flags. Combining `--config`
  with these flags is rejected.
- **Preprocessing report** (`reports.py`): `PreprocessReport` driven by
  `PreprocessConfig` + `DatasetProfile`. Rich terminal display + `report.json`
  export.
- **Pipeline safety** (`transform` command): `.joblib` extension check,
  `ColumnTransformer` type check, missing required columns → hard error,
  extra columns → warn + ignore, column order independence.
- **Test suite expansion**: `test_config.py`, `test_strategies.py`,
  `test_reports.py`, `test_pipeline_safety.py` (73 tests total).
- **`examples/config.yaml`**: documented YAML template for users.

### Changed

- `pipeline.py`: engine rewritten to consume `PreprocessConfig`.
  `create_preprocessor(cols, config)` — no more bare strategy params.
- `cli.py`: unified all modes into a single flow via `_resolve_config()`.
- `pipeline.py`: fixed Pandas 3 deprecation — `object` dtype → `str` in
  `select_dtypes`.

### Fixed

- Typo: `numerical_stratergy` → `numerical_strategy` throughout.

---

## [0.1.0] — 2026-09-08

### Added (V1 — Foundation)

- `mlprep inspect` command: profile a CSV dataset (column types, missing values).
- `mlprep preprocess` command: impute, encode, scale, and export a dataset.
- `mlprep transform` command: apply a saved pipeline to new data.
- `DatasetProfile` model and `profile_dataset()`.
- `load_dataset()` and `load_validated_dataset()` in `io.py`.
- `validate_input()` in `validator.py`.
- Initial test suite: `test_pipeline.py`, `test_profiler.py`, `test_cli.py`.
