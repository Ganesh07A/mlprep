# MLPrep — Master Project Plan

> A production-style CLI tool for inspecting datasets, applying reproducible machine-learning preprocessing, and exporting reusable preprocessing pipelines.

## 1. Project Goal

MLPrep should make common ML preprocessing faster, safer, and reproducible.

The final user experience should support three usage styles:

```text
Beginner
    ↓
Interactive preprocessing wizard

Expert developer
    ↓
CLI options

Automation / reproducibility
    ↓
YAML configuration
```

Core workflow:

```text
Dataset
   ↓
Validate
   ↓
Inspect / Profile
   ↓
Choose preprocessing
   ↓
Build sklearn pipeline
   ↓
Fit + Transform training data
   ↓
Save processed dataset
   ↓
Save fitted preprocessing pipeline
   ↓
Reuse pipeline on new data
```

---

## 2. Current Status

### Completed

- [x] `uv` project setup
- [x] Typer CLI
- [x] `inspect` command
- [x] `preprocess` command
- [x] `transform` command
- [x] CSV validation and loading
- [x] Dataset profiling
- [x] Missing-value detection and percentages
- [x] Numerical/categorical detection
- [x] Rich terminal output
- [x] `DatasetProfile` dataclass
- [x] Numerical imputation
- [x] Categorical imputation
- [x] One-hot encoding
- [x] Standard scaling
- [x] `Pipeline` + `ColumnTransformer`
- [x] Processed CSV export
- [x] Fitted `preprocessor.joblib` export
- [x] Reuse of saved pipeline with `transform()`
- [x] Output-directory creation
- [x] Overwrite protection
- [x] Core pytest tests
- [x] CLI tests
- [x] Ruff cleanup
- [x] Refactoring of duplicated preprocessing/dataframe logic
- [x] Initial interactive numerical-strategy selector

### Current design direction

```bash
# Interactive mode
mlprep preprocess data.csv

# Expert mode
mlprep preprocess data.csv --strategy median --scaler standard

# Reproducible config mode
mlprep preprocess data.csv --config config.yaml

# Reuse an existing fitted pipeline
mlprep transform new_data.csv data/preprocessor.joblib
```

---

# 3. Version Roadmap

```text
V1 — Foundation
    ↓
V2 — Core ML Tool
    ↓
V3 — Production / Portfolio Ready
```

---

# VERSION 1 — FOUNDATION

## Objective

Build the core CLI and dataset inspection capabilities.

## Status

Mostly complete.

### V1.1 Project Setup

- [x] Initialize project with `uv`
- [x] Create `pyproject.toml`
- [x] Add Typer
- [x] Configure package entry point
- [x] Create `src/mlprep` package

### V1.2 CLI Foundation

- [x] Create Typer application
- [x] Register multiple commands
- [x] `inspect`
- [x] `preprocess`
- [x] `transform`

Target:

```bash
mlprep --help
mlprep inspect data.csv
mlprep preprocess data.csv
mlprep transform data.csv pipeline.joblib
```

### V1.3 Validation and Loading

Files:

```text
src/mlprep/
├── validator.py
└── io.py
```

Tasks:

- [x] Validate file existence
- [x] Validate supported extension
- [x] Load CSV with Pandas
- [x] Handle empty files
- [x] Handle malformed CSV
- [x] Handle decoding errors
- [x] Reusable `load_validated_dataset()`

### V1.4 Dataset Profiling

File:

```text
src/mlprep/profiler.py
```

Tasks:

- [x] Row count
- [x] Column count
- [x] Data types
- [x] Numerical columns
- [x] Categorical columns
- [x] Missing-value count
- [x] Missing percentage
- [x] Structured profile
- [x] `DatasetProfile` dataclass

### V1.5 Display

File:

```text
src/mlprep/display.py
```

Tasks:

- [x] Rich output
- [x] Dataset summary
- [x] Missing-value table
- [x] Data-type table
- [x] Separate display from profiling

### V1.6 V1 Tests

- [x] Profiler tests
- [x] Missing-value tests
- [x] Numerical/categorical detection tests
- [x] CLI inspect test
- [x] Invalid-file test

---

# VERSION 2 — CORE ML TOOL

## Objective

Turn MLPrep into a configurable, reusable ML preprocessing tool.

This is the key portfolio version.

## V2.1 Preprocessing Engine

Files:

```text
src/mlprep/preprocess/
├── missing.py
├── encoding.py
├── scaling.py
└── pipeline.py
```

Tasks:

- [x] Missing-value handling
- [x] One-hot encoding
- [x] Standard scaling
- [x] sklearn `Pipeline`
- [x] `ColumnTransformer`
- [x] `preprocess_dataset()`
- [x] `transform_dataset()`
- [x] `create_processed_dataframe()`
- [x] `get_column_types()`

Architecture:

```text
                    Raw Data
                       |
               Column classification
                       |
           +-----------+-----------+
           |                       |
      Numerical                Categorical
           |                       |
       Imputer                  Imputer
           |                       |
        Scaler                  Encoder
           |                       |
           +-----------+-----------+
                       |
                Processed Data
```

## V2.2 Missing-Value Strategies

Current defaults:

```text
Numerical → median
Categorical → most_frequent
```

Build:

```text
Numerical:
- median
- mean
- constant

Categorical:
- most_frequent
- constant
```

Tasks:

- [ ] Define supported strategies
- [ ] Validate strategy names
- [x] Numerical strategy parameter
- [ ] Categorical strategy parameter
- [ ] Constant-value input
- [ ] Unit tests for every strategy

## V2.3 Interactive Wizard

File:

```text
src/mlprep/ui.py
```

Dependency:

```bash
uv add questionary
```

Tasks:

- [x] Numerical strategy selector
- [ ] Categorical strategy selector
- [ ] Encoding selector
- [ ] Scaling selector
- [ ] Review screen
- [ ] Confirmation before running
- [ ] Display selected configuration

Target:

```text
MLPrep Preprocessing

Dataset: train.csv

1. Numerical missing values
   > Median
     Mean
     Constant

2. Categorical missing values
   > Most Frequent
     Constant

3. Encoding
   > One-Hot

4. Scaling
   > Standard

Apply preprocessing? [Y/n]
```

## V2.4 Expert CLI Mode

Goal: experts should not be forced into the wizard.

Example:

```bash
mlprep preprocess data.csv     --strategy median     --scaler standard     --encoder onehot
```

Tasks:

- [ ] Add CLI options
- [ ] Validate option combinations
- [ ] Detect explicit options
- [ ] Explicit options → expert mode
- [ ] No options → interactive mode
- [ ] Keep both modes backed by the same engine

Decision flow:

```text
User runs preprocess
        |
        v
CLI options supplied?
   /               yes              no
  |                |
Expert mode    Interactive mode
  |                |
  +-------+--------+
          |
     Same pipeline
```

## V2.5 YAML Configuration

Dependency:

```bash
uv add pyyaml
```

Example:

```yaml
missing_values:
  numerical: median
  categorical: most_frequent

encoding:
  method: onehot

scaling:
  method: standard
```

CLI:

```bash
mlprep preprocess data.csv --config config.yaml
```

Tasks:

- [ ] Add YAML dependency
- [ ] Create configuration model
- [ ] Load YAML
- [ ] Validate YAML
- [ ] Apply defaults
- [ ] Map config to preprocessing engine
- [ ] Handle invalid config errors
- [ ] Add config tests

Important:

```text
Equivalent YAML and CLI settings
            ↓
      same pipeline
            ↓
       same result
```

## V2.6 Preprocessing Report

Goal: tell the user exactly what happened.

Target:

```text
Preprocessing Report
────────────────────────────

Input:
train.csv

Rows:
10,000

Numerical columns:
8

Categorical columns:
4

Transformations:
age              → median imputation + standard scaling
salary           → median imputation + standard scaling
city             → most-frequent imputation + one-hot encoding

Output:
processed.csv

Pipeline:
preprocessor.joblib
```

Tasks:

- [ ] Report model
- [ ] Record selected strategies
- [ ] Record transformed columns
- [ ] Record outputs
- [ ] Rich terminal report
- [ ] JSON report
- [ ] Report tests

## V2.7 Saved Pipeline / Transform Workflow

Current:

- [x] Save fitted pipeline
- [x] Load fitted pipeline
- [x] Transform new data

Improve:

- [ ] Validate `.joblib`
- [ ] Handle incompatible pipeline
- [ ] Validate expected columns
- [ ] Handle missing expected columns
- [ ] Handle extra columns
- [ ] Provide useful compatibility errors

Workflow:

```text
Training:
mlprep preprocess train.csv

Outputs:
processed.csv
preprocessor.joblib
report.json

New data:
mlprep transform new_data.csv preprocessor.joblib
```

## V2.8 Pipeline Safety

Tasks:

- [x] Fit only during training
- [x] Transform without refitting
- [x] Unknown-category test
- [ ] Column-order tests
- [ ] Missing-required-column test
- [ ] Extra-column test
- [ ] Empty-input test
- [ ] One-row test

Rule:

```text
Training data
    ↓
fit()
    ↓
learn preprocessing rules
    ↓
save pipeline

New data
    ↓
transform()
    ↓
reuse learned rules
```

## V2.9 V2 Testing

Current:

- [x] Profiler tests
- [x] Pipeline missing-value test
- [x] Encoding test
- [x] Unknown-category test
- [x] Scaling test
- [x] CLI inspect test
- [x] CLI preprocess test
- [x] CLI transform test
- [x] Saved-pipeline test
- [x] Reuse-pipeline test

Add:

- [ ] Strategy tests
- [ ] Configuration tests
- [ ] Report tests
- [ ] Error-path tests
- [ ] Output-file tests
- [ ] Pipeline compatibility tests

Testing rule:

```text
Important feature
    ↓
success case
+
failure / edge case
```

---

# VERSION 3 — PRODUCTION / PORTFOLIO READY

## Objective

Make MLPrep feel like a serious open-source developer tool.

## V3.1 CLI UX

Tasks:

- [ ] Better command help
- [ ] Better option descriptions
- [ ] Consistent terminal messages
- [ ] Useful progress/status output
- [ ] Clear success messages
- [ ] Clear warnings
- [ ] Friendly errors
- [ ] Document exit codes

Example:

```text
✓ Dataset loaded
✓ Profile generated
✓ Missing values handled
✓ Categorical features encoded
✓ Numerical features scaled
✓ Processed dataset saved
✓ Pipeline saved
```

## V3.2 Error Handling

Handle:

- [ ] Missing input file
- [ ] Unsupported extension
- [ ] Empty dataset
- [ ] Invalid CSV
- [ ] Invalid configuration
- [ ] Invalid strategy
- [ ] Invalid pipeline
- [ ] Missing required columns
- [ ] Incompatible dataset
- [ ] Permission errors
- [ ] Output-write failures

Design:

```text
Internal exception
       ↓
CLI boundary
       ↓
Human-readable message
       ↓
Correct exit code
```

## V3.3 Logging

Tasks:

- [ ] Python logging
- [ ] INFO/WARNING/ERROR/DEBUG
- [ ] Optional `--verbose`
- [ ] Keep normal terminal output clean

Example:

```bash
mlprep preprocess data.csv --verbose
```

## V3.4 Data Quality Checks

Warn about:

- [ ] High missingness
- [ ] Constant columns
- [ ] Duplicate rows
- [ ] Duplicate column names
- [ ] High-cardinality columns
- [ ] Unsupported data types
- [ ] Potential datetime columns
- [ ] Identifier-like columns
- [ ] Highly imbalanced categorical features

Use warnings/recommendations rather than silent destructive changes.

Example:

```text
⚠ customer_id looks identifier-like.
  Recommendation: review before encoding.
```

## V3.5 Feature Recommendations

Do not automatically delete suspicious columns.

Signals may include:

- column name
- uniqueness ratio
- data type
- number of distinct values
- ID-like patterns

Output:

```text
Potential issue detected:

customer_id
- 100% unique
- identifier-like name

Recommendation:
Consider excluding this column from ML features.
```

## V3.6 File Support

Only after CSV is stable.

Potential formats:

- [ ] CSV
- [ ] XLSX
- [ ] Parquet

Architecture:

```text
Input file
   ↓
format detection
   ↓
appropriate loader
   ↓
DataFrame
```

## V3.7 Machine-Readable Output

Example:

```bash
mlprep inspect data.csv --format json
```

Output:

```json
{
  "rows": 1000,
  "columns": 12,
  "numerical_columns": 8,
  "categorical_columns": 4
}
```

Use case:

```text
MLPrep
  ↓
automation / CI / scripts
```

## V3.8 GitHub Actions CI

Create:

```text
.github/
└── workflows/
    └── ci.yml
```

Workflow:

```text
GitHub push / PR
       ↓
Install Python
       ↓
Install dependencies with uv
       ↓
Ruff check
       ↓
Pytest
       ↓
PASS / FAIL
```

Tasks:

- [ ] Create CI workflow
- [ ] Install dependencies via uv
- [ ] Run Ruff
- [ ] Run pytest
- [ ] Test supported Python versions
- [ ] Optionally require CI before merge

## V3.9 Pre-commit

Optional.

```bash
uv add --dev pre-commit
```

Tasks:

- [ ] Configure pre-commit
- [ ] Run Ruff automatically
- [ ] Run fast tests
- [ ] Document setup

## V3.10 Documentation

README should include:

### Overview
What problem MLPrep solves.

### Features

```text
✓ Dataset profiling
✓ Missing-value handling
✓ Categorical encoding
✓ Feature scaling
✓ Reproducible pipelines
✓ Interactive mode
✓ Expert CLI mode
✓ YAML configuration
```

### Installation

```bash
uv sync
```

### Quick Start

```bash
mlprep inspect data.csv
mlprep preprocess data.csv
mlprep transform new_data.csv preprocessor.joblib
```

### Interactive Mode

Show terminal examples or GIF/screenshots.

### Expert Mode

Show CLI options.

### Configuration

Show YAML configuration.

### Architecture

Add architecture diagram.

### Testing

```bash
uv run pytest
```

### Development

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

### Limitations

Document unsupported formats and advanced ML cases honestly.

### Roadmap

Show future improvements.

## V3.11 Packaging

Goal: make installation easy.

Tasks:

- [ ] Verify package metadata
- [ ] Define supported Python versions
- [ ] Verify build configuration
- [ ] Build wheel
- [ ] Test installed package
- [ ] Verify CLI entry point
- [ ] Prepare release workflow

## V3.12 Release

Tasks:

- [ ] Semantic versioning
- [ ] Changelog
- [ ] GitHub release
- [ ] Git tag
- [ ] Version information
- [ ] Final installation documentation

---

# 4. Recommended Final Structure

```text
mlprep/
│
├── src/
│   └── mlprep/
│       ├── __init__.py
│       ├── cli.py
│       ├── io.py
│       ├── validator.py
│       ├── models.py
│       ├── profiler.py
│       ├── display.py
│       ├── ui.py
│       ├── config.py
│       ├── reports.py
│       ├── logging_config.py
│       │
│       └── preprocess/
│           ├── __init__.py
│           ├── missing.py
│           ├── encoding.py
│           ├── scaling.py
│           └── pipeline.py
│
├── tests/
│   ├── test_cli.py
│   ├── test_profiler.py
│   ├── test_pipeline.py
│   ├── test_config.py
│   ├── test_reports.py
│   └── fixtures/
│
├── data/
│   └── sample.csv
│
├── examples/
│   └── config.yaml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore
├── README.md
├── LICENSE
├── pyproject.toml
└── uv.lock
```

Do not create all of these files at once. Introduce files when their responsibility is needed.

---

# 5. Dependency Plan

Current/core:

```text
Python
uv
Typer
Pandas
NumPy
Scikit-learn
Rich
joblib
pytest
Ruff
```

Add when needed:

```text
Questionary → interactive selections
PyYAML      → configuration
pre-commit  → developer workflow
```

Avoid dependencies without a concrete feature requiring them.

---

# 6. Architecture Rules

## Rule 1 — CLI is orchestration

`cli.py` should coordinate modules rather than contain all business logic.

```text
cli.py
   ↓
specialized modules
```

## Rule 2 — UI is separate from preprocessing

```text
ui.py
   ↓
user choice

pipeline.py
   ↓
actual ML transformation
```

The same core engine should work with:

- interactive mode
- expert CLI mode
- YAML config
- automated scripts
- tests

## Rule 3 — Never silently refit new data

Training:

```text
fit_transform()
```

New data:

```text
transform()
```

## Rule 4 — Prefer recommendations over destructive automation

Don't silently delete or rewrite suspicious data.

## Rule 5 — Test behavior, not implementation details

Prefer:

```python
assert profile.missing_count["age"] == 1
```

rather than testing internal variables.

---

# 7. Definition of Done

MLPrep is portfolio-ready when the following are complete.

## Functionality

- [ ] Interactive preprocessing works
- [ ] Expert CLI options work
- [ ] YAML configuration works
- [ ] Inspect works
- [ ] Preprocess works
- [ ] Transform works
- [ ] Processed data exports
- [ ] Pipeline exports
- [ ] Pipeline can be reused
- [ ] Reports are generated

## Reliability

- [ ] Invalid inputs handled
- [ ] Edge cases handled
- [ ] Pipeline compatibility checked
- [ ] No accidental refitting
- [ ] Important behaviors tested

## Code Quality

- [ ] Ruff clean
- [ ] Tests pass
- [ ] Clear project structure
- [ ] Clear responsibilities
- [ ] No unnecessary duplication
- [ ] Useful type hints

## DevOps

- [ ] GitHub Actions CI
- [ ] CI runs pytest
- [ ] CI runs Ruff
- [ ] Build/install verified

## Documentation

- [ ] Complete README
- [ ] Installation instructions
- [ ] Usage examples
- [ ] Architecture diagram
- [ ] Config example
- [ ] Testing instructions
- [ ] Limitations documented

## Portfolio Presentation

- [ ] Strong project description
- [ ] Screenshots/GIF
- [ ] Polished GitHub repo
- [ ] Meaningful commit history
- [ ] Resume bullet
- [ ] Demo dataset
- [ ] Example workflows

---

# 8. Final Portfolio Positioning

Avoid:

> A Python CLI that cleans datasets.

Prefer:

> **MLPrep is a reproducible machine-learning preprocessing CLI that profiles datasets, applies configurable preprocessing pipelines, generates processed datasets and reusable scikit-learn pipelines, and supports interactive and automation-friendly workflows.**

It should demonstrate:

```text
Python
+
Data Engineering
+
Machine Learning
+
Scikit-learn
+
CLI Development
+
Software Architecture
+
Testing
+
CI/CD
+
Reproducibility
```

---

# 9. Execution Method

For each feature:

```text
1. Understand the problem
2. Understand why it exists
3. Design the smallest solution
4. Implement it
5. Run it manually
6. Add/update tests
7. Run Ruff
8. Commit the change
9. Move on
```

Avoid building unrelated features simultaneously.

---

# 10. Immediate Next Route

```text
CURRENT
  ↓
Finish interactive strategy system
  ↓
Add expert CLI options
  ↓
Add categorical strategy selection
  ↓
Add scaling selection
  ↓
Add encoding selection
  ↓
Add configuration model
  ↓
Add YAML config
  ↓
Add preprocessing report
  ↓
Strengthen pipeline validation
  ↓
Expand test suite
  ↓
GitHub Actions CI
  ↓
Documentation
  ↓
Packaging
  ↓
Release
  ↓
MLPrep v1.0
```

---

# 11. Milestones

## Milestone A — Core Engine

```text
Dataset
→ inspect
→ preprocess
→ transform
→ export
→ saved pipeline
```

Status: **Completed / stabilized**

## Milestone B — Flexible UX

```text
Interactive mode
+
Expert options
+
YAML configuration
```

Status: **Next major milestone**

## Milestone C — Production Quality

```text
Reports
+
Validation
+
Testing
+
CI
+
Documentation
+
Packaging
```

Status: **Final milestone**

---

# 12. Success Metric

A developer should be able to clone the repository and understand the workflow quickly:

```bash
uv sync

uv run mlprep inspect data.csv

uv run mlprep preprocess data.csv

uv run mlprep transform new_data.csv data/preprocessor.joblib
```

An ML practitioner should also be able to automate it:

```bash
mlprep preprocess data.csv --config config.yaml
```

The combination of **ease of use + reproducibility + engineering quality** is what turns MLPrep from a learning exercise into a strong portfolio project.
