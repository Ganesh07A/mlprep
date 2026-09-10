# MLPrep

> **Reproducible ML preprocessing CLI** — profile datasets, apply configurable
> preprocessing pipelines, and export reusable scikit-learn pipelines.

[![CI](https://github.com/Ganesh07A/mlprep/actions/workflows/ci.yml/badge.svg)](https://github.com/Ganesh07A/mlprep/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What problem does it solve?

Every ML project starts with the same manual steps:

- Load a CSV
- Check for missing values
- Impute, encode, and scale features
- Fit a pipeline on training data
- Apply that *same* pipeline to new data — without refitting

MLPrep automates all of this from the command line, with three usage modes
suitable for beginners, expert practitioners, and fully automated pipelines.

---

## Features

| Feature | Status |
|---|---|
| Dataset profiling (column types, missing values) | ✅ |
| Data quality warnings (constant cols, ID-like cols, high missingness, …) | ✅ |
| Numerical imputation (median / mean / constant) | ✅ |
| Categorical imputation (most_frequent / constant) | ✅ |
| One-hot encoding | ✅ |
| Standard scaling | ✅ |
| Reproducible sklearn pipelines | ✅ |
| Interactive wizard | ✅ |
| Expert CLI flags | ✅ |
| YAML configuration | ✅ |
| Machine-readable JSON output | ✅ |
| Preprocessing report | ✅ |
| Column order independence at transform time | ✅ |
| GitHub Actions CI | ✅ |

---

## Architecture

```
           Interactive Wizard ─┐
           Expert CLI flags ───┼──→ PreprocessConfig
           YAML --config ──────┘          │
                                          ▼
                               ┌─────────────────────┐
                               │   Pipeline Engine   │
                               │ ColumnTransformer   │
                               └────────┬────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                             ▼
                   processed.csv              preprocessor.joblib
                                                        │
                                           New data ────┘
                                                        ▼
                                              transform (no refit)
                                                        │
                                                transformed.csv
```

---

## Installation

```bash
# Clone and install with uv
git clone https://github.com/Ganesh07A/mlprep.git
cd mlprep
uv sync
```

---

## Quick Start

```bash
# 1. Inspect your dataset
uv run mlprep inspect data/sample.csv

# 2. Preprocess it
uv run mlprep preprocess data/sample.csv --output data/processed.csv

# 3. Apply the saved pipeline to new data
uv run mlprep transform new_data.csv data/preprocessor.joblib
```

---

## Usage Modes

### Interactive Wizard

Run `preprocess` without any flags in a real terminal:

```bash
uv run mlprep preprocess data.csv
```

```
MLPrep — Preprocessing Wizard

? How should numerical missing values be handled?
  ❯ median
    mean
    constant

? How should categorical missing values be handled?
  ❯ most_frequent
    constant

Encoding:  ✓ One-Hot Encoding
Scaling:   ✓ Standard Scaling

? Apply preprocessing with these settings? [Y/n]
```

### Expert CLI Mode

Skip the wizard with explicit flags:

```bash
uv run mlprep preprocess data.csv \
  --strategy mean \
  --cat-strategy constant \
  --output results/clean.csv
```

### YAML Configuration Mode

Define preprocessing once, reuse everywhere:

```bash
uv run mlprep preprocess data.csv --config config.yaml
```

```yaml
# config.yaml
missing_values:
  numerical: median
  categorical: most_frequent

encoding:
  method: onehot

scaling:
  method: standard
```

> **Note:** `--config` and `--strategy` / `--cat-strategy` cannot be combined.
> Use one interface at a time.

### Machine-Readable JSON Output

```bash
uv run mlprep inspect data.csv --format json
```

```json
{
  "rows": 1000,
  "columns": 12,
  "numerical_columns": ["age", "salary"],
  "categorical_columns": ["city", "department"],
  "missing": {
    "age": { "count": 42, "percentage": 4.2 }
  }
}
```

---

## Data Quality Warnings

MLPrep automatically detects common ML anti-patterns and warns you before
they silently hurt your model:

| Check | Condition |
|---|---|
| High missingness | > 50% of values missing |
| Constant column | Only 1 unique value |
| Duplicate rows | Any duplicated rows |
| High cardinality | > 50 unique values in a categorical column |
| Identifier-like column | ID-like name + high uniqueness |
| Potential datetime | Object column whose values parse as dates |
| Highly imbalanced categorical | Dominant class > 95% |

Warnings are **recommendations only** — data is never silently modified.

---

## Outputs

Running `preprocess` creates three files next to `--output`:

| File | Content |
|---|---|
| `processed.csv` | Transformed dataset |
| `preprocessor.joblib` | Fitted sklearn ColumnTransformer |
| `report.json` | Machine-readable preprocessing report |

---

## Security Note

> Only load `preprocessor.joblib` files generated by **trusted MLPrep runs**.
> Python's `joblib`/`pickle` can execute arbitrary code in tampered files.

---

## Testing

```bash
uv run pytest -q
```

---

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Tests
uv run pytest -q
```

---

## Limitations

- Only CSV input is supported (XLSX/Parquet planned for a future version)
- Encoding is limited to one-hot (target encoding planned)
- Scaling is limited to standard scaling (min-max planned)
- No automatic feature selection or dimensionality reduction
- No support for time-series or text features

---

## Roadmap

- [ ] XLSX and Parquet support
- [ ] Target encoding and frequency encoding
- [ ] Min-max and robust scaling
- [ ] `--exclude-columns` flag
- [ ] `mlprep validate` command
- [ ] PyPI release

---

## License

MIT
