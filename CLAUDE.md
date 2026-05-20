# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gaia applies causal inference to U.S. crop return volatility (corn, soybean, wheat), testing causal dependencies between realized volatility and macroeconomic indicators, news sentiment, and climate variables using multivariate time-series methods (PCMCI, CMI, Gaussian Processes, partial correlation).

## Environment Setup

This project uses **conda**, not pip/uv. There are three environment files:

```bash
# CPU / development (default)
conda env create -f environment.yml        # creates env named "gaia"
conda activate gaia

# CUDA GPU (Linux + NVIDIA)
conda env create -f environment.cuda.yml   # creates env named "gaia-cuda"

# Add dev tools (JupyterLab, ruff) on top of an existing env
conda env update -f environment.dev.yml
```

## Common Commands

```bash
# Run all PCMCI configs under config/
python scripts/pcmci.py

# Run a single config file
python scripts/pcmci.py --config config/corn/pcmci/parcorr/some_config.json

# Filter by dependence method or data frequency
python scripts/pcmci.py --method parcorr
python scripts/pcmci.py --frequency weekly
python scripts/pcmci.py --method cmi --frequency monthly

# Control log verbosity
python scripts/pcmci.py --log-level DEBUG

# Lint
ruff check .
ruff format .
```

## Architecture

### Data Flow

```
config/*.json → load_config() → RunConfig
                                    ↓
                              load_dataset() → select_feature() → prepare_numeric_frame()
                                    ↓
                              tigramite DataFrame
                                    ↓
                         build_dependence_test() → PCMCI.run_pcmci()
                                    ↓
                              extract_links() → RunResult
                                    ↓
                              save_outputs() → benchmark/<freq>/<crop>/pcmci/<method>/<run_name>/
```

### Key Modules

- **`src/config/model.py`** — All dataclasses: `RunConfig`, `DataConfig`, `FeatureSelectionConfig`, `DependenceConfig`, `PCMCIConfig`, `OutputConfig`, `RunResult`, `LinkResult`.
- **`src/config/load.py`** — Deserializes JSON configs into `RunConfig`. `load_config_dir()` globs a directory recursively.
- **`src/dataset/feature.py`** — Defines `FEATURE_GROUPS` (the canonical column-name registry for `rv`, `macro`, `news`, `climate`, `climate_raw`). The `rv` group uses the sentinel `__weekly_rv__` which is resolved at runtime to any `*_weekly_rv` column in the dataset.
- **`src/dataset/data.py`** — `load_dataset()` (CSV → DataFrame) and `prepare_numeric_frame()` (drops non-numeric and NaN rows).
- **`src/pcmci/dependence.py`** — Factory that maps method name → tigramite `CondIndTest`. Supported methods: `parcorr`, `cmi`, `gpdc_torch`.
- **`src/pcmci/cmi.py`** — Custom `FAISSCMI` class: a kNN-based Conditional Mutual Information test backed by FAISS (CPU or GPU). Handles SWIG pickle issues by rebuilding GPU resources on `__setstate__`. Falls back to CPU automatically if GPU is unavailable.
- **`src/pcmci/runner.py`** — `run_pcmci()` orchestrates the full pipeline; `save_outputs()` writes `summary.json`, `links.csv`, `graph.txt`, and optional PNG/HTML plots to `benchmark/`.
- **`src/pcmci/graph.py`** — Tigramite and NetworkX/pyvis visualization helpers.

### Config Structure

Configs live at `config/<crop>/pcmci/<method>/<name>.json`. Output mirrors this under `benchmark/<frequency>/<crop>/pcmci/<method>/<run_name>/`.

Each JSON config keys: `name`, `data` (path, date_column, drop_columns), `features` (include_groups, include_columns, exclude_columns), `dependence` (method, params), `pcmci` (tau_min/max, pc_alpha, alpha_level, fdr_method, etc.), `output` (directory, run_name, max_links, save_tigramite_plots, save_networkx_plot).

### Data

- `data/<frequency>/<crop>.csv` — processed datasets (features engineered)
- `data/<frequency>/<crop>_raw.csv` — raw climate variables (prcp, awnd, tmin, tmax, co2, pdsi)
- Managed via DVC (`data.dvc`); pull with `dvc pull`

## Linting

`ruff.toml` uses `select = ["ALL"]` with targeted ignores. Line length is 92. Notebooks and `src/dataset/news/bq_query.py` are excluded. Run `ruff check --fix .` to auto-fix.
