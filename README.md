# ExPO: Exposure–Response Neural Operator for Gene Expression Imputation

This repository contains a reference implementation of **ExPO**, a neural-operator–style model for learning dose–time–conditioned gene expression responses to small-molecule perturbations. The codebase is structured as a small research framework:

- Data loading and preprocessing for perturbational transcriptomics–style datasets  
- A modular ExPO model combining a chemical encoder and exposure (dose/time) encoder  
- Training, evaluation, and calibration utilities  
- Plotting and reporting utilities  
- A self-contained synthetic data generator and demo pipeline

The intent is to provide a reproducible and inspectable implementation that matches the methodology described in the accompanying manuscript, while remaining flexible enough to support related experiments.

---

## 1. High-Level Overview

ExPO treats gene expression as a function of:

- **Compound structure** (via a sequence-based chemical encoder)
- **Exposure conditions** (dose, time; optionally encoded via Fourier features and/or raw scalars)
- **Context variables** (e.g., cell line indices)

Given these inputs, the model predicts **full expression profiles** and supports:

- Regression metrics (MAE, RMSE, R², Pearson)
- Direction-of-change classification (via thresholds)
- Ranking metrics over top-K responsive genes
- Uncertainty estimation and calibration (quantile and conformal methods)

The repository is organized so that:

- Core model and utilities live under `expo/`
- Scripts for end-to-end experiments live under `scripts/`
- Configuration is JSON/YAML-style, making runs easy to reproduce

A demo pipeline is included that can be run end-to-end on **synthetic data** to verify that the environment and code are functioning correctly.

---

## 2. Repository Structure

A non-exhaustive overview of key modules:

```text
ExPO/
├─ expo/
│  ├─ __init__.py                 # Package entry: exposes main model & utilities
│  ├─ config.py                   # ExperimentConfig and configuration helpers
│  ├─ data/
│  │   ├─ datasets.py             # Dataset objects, loaders, split logic
│  │   └─ transforms.py           # Preprocessing & feature transforms
│  ├─ models/
│  │   ├─ __init__.py             # Model registry
│  │   ├─ expo_model.py           # Main ExPO model (chem + exposure + readout)
│  │   ├─ chem_encoder.py         # ChemBERTa-style encoder wrapper
│  │   └─ exposure_encoder.py     # Dose/time encodings, Fourier features, etc.
│  ├─ metrics/
│  │   ├─ __init__.py
│  │   ├─ regression_metrics.py   # MAE, RMSE, R², Pearson, per-gene metrics
│  │   ├─ ranking_metrics.py      # NDCG@K, Jaccard@K, RBO, etc.
│  │   └─ uncertainty_metrics.py  # Interval score, coverage, risk–coverage curves
│  ├─ calibration/
│  │   ├─ __init__.py
│  │   ├─ utils.py                # Shared utilities + CalibrationResult container
│  │   ├─ quantile_calibration.py # Quantile scaling, groupwise calibration
│  │   ├─ conformal.py            # Split/multi-output conformal regression
│  │   ├─ temperature_scaling.py  # Logit temperature scaling for classification
│  │   ├─ reliability.py          # ECE, RMSE calibration error, reliability curves
│  │   └─ multioutput.py          # Per-dimension/vector calibrators
│  ├─ baselines/
│  │   ├─ __init__.py
│  │   └─ wrappers.py             # Constant, identity, per-cell mean, dose-scaled baselines
│  ├─ training/
│  │   ├─ __init__.py
│  │   ├─ trainer.py              # Training loop, validation, early stopping
│  │   └─ logging.py              # JSONL logger, log loading utilities
│  ├─ reporting/
│  │   ├─ __init__.py
│  │   ├─ tables.py               # Convert metrics dicts to DataFrames, comparison tables
│  │   └─ markdown.py             # DataFrame → markdown, report generator
│  ├─ utils/
│  │   ├─ __init__.py
│  │   ├─ typing_utils.py         # Path/array helpers & protocols
│  │   ├─ seed.py                 # Reproducible seeding across numpy/torch
│  │   └─ misc.py                 # Small convenience utilities
│  └─ constants.py                # Default thresholds, landmark counts, metric configs
│
├─ scripts/
│  ├─ generate_test_data.py       # Structured synthetic dataset generator
│  ├─ train_expo.py               # Main training entry point
│  ├─ plot_training_curves.py     # Training curve & gap visualizations
│  ├─ plot_uncertainty_curves.py  # Calibration & risk–coverage plots
│  ├─ run_full_demo_pipeline.py   # End-to-end demo: generate → train → plot → cleanup
│  └─ install_dependencies.py     # Helper for installing Python dependencies
│
├─ transformers/                  # (Optional) local stub to avoid installing real HF
│  └─ __init__.py
│
├─ requirements.txt               # Python dependencies
└─ README.md                      # This file
```

> **Note:** In a production or benchmarking setting, you would typically use the actual `transformers` library (and a pre-trained chemistry model). A lightweight local stub is included primarily to make the demo pipeline easy to run in constrained environments.

---

## 3. Installation

### 3.1. Clone and move into the project

```bash
git clone <your-repo-url> ExPO
cd ExPO
```

On Windows PowerShell, this is:

```powershell
git clone <your-repo-url> ExPO
cd ExPO
```

### 3.2. (Recommended) Create a virtual environment and install dependencies

Use the helper script:

```bash
# From the project root
python scripts/install_dependencies.py --create-venv --upgrade-pip
```

This will:

1. Create a virtualenv at `.venv/`
2. Upgrade `pip`
3. Install all packages from `requirements.txt`

Activate the environment:

- **Windows (PowerShell)**  
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- **Linux/macOS (bash/zsh)**  
  ```bash
  source .venv/bin/activate
  ```

### 3.3. Ensure the project is importable

From the project root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

(or on Linux/macOS: `export PYTHONPATH=$(pwd)`)

This lets Python find the `expo` package and the local `transformers` stub (if used).

> **Alternative:** You can also install the project as an editable package via `pip install -e .` once a lightweight `pyproject.toml` / `setup.cfg` is added.

### 3.4. Using real vs. stub transformers

- To **use the local stub**: keep the `transformers/` folder in the repo, do *not* install `transformers` via pip.
- To **use the real HuggingFace Transformers** (recommended for serious experimentation):

  ```bash
  pip install transformers
  ```

  and remove or rename the local `transformers/` directory so the real package is used.

---

## 4. Quickstart: End-to-End Demo Pipeline

To verify everything works, run the full demo on synthetic data:

```bash
# From the repo root, with venv activated and PYTHONPATH set
python scripts/run_full_demo_pipeline.py
```

The pipeline performs:

1. **Synthetic data generation** (`generate_test_data.py`)  
   - Writes `expression_table.pkl`, `metadata_table.pkl`, `compound_table.pkl` under `demo_workspace/data/`
   - Saves a `generation_config.json` describing the synthetic generation parameters

2. **Configuration creation**  
   - Writes `demo_workspace/configs/demo_config.json`, an `ExperimentConfig` compatible JSON pointing to the synthetic data

3. **Model training** (`train_expo.py`)  
   - Trains ExPO for a small number of epochs
   - Logs metrics to `runs/full_demo/metrics.jsonl`
   - Saves the best checkpoint under `runs/full_demo/`

4. **Plotting** (`plot_training_curves.py`)  
   - Produces training/validation curves and auxiliary plots in `runs/full_demo/figures/`

By default, the pipeline **removes the synthetic data directory** at the end to keep the workspace tidy. To keep the generated data:

```bash
python scripts/run_full_demo_pipeline.py --keep-data
```

You can also control synthetic dataset size:

```bash
python scripts/run_full_demo_pipeline.py --n-profiles 1000 --n-genes 300
```

---

## 5. Using Your Own Data

The expected input format mirrors typical perturbational gene expression resources:

### 5.1. Required tables

1. **Expression table** (`expression_table.pkl`)

   - Format: `pandas.DataFrame` pickled with `.to_pickle(...)`
   - Columns:
     - `profile_id` (integer or string ID)
     - One column per gene (e.g., `G0000`, `G0001`, … or gene symbols)

2. **Metadata table** (`metadata_table.pkl`)

   - Format: `pandas.DataFrame`
   - Columns (recommended):
     - `profile_id` (to join with expression table)
     - `compound_id` (string ID linking to compound table)
     - `cell_id` (cell line identifier or index)
     - `time` (exposure time, in hours)
     - `dose` (exposure dose, in consistent units, e.g., µM)

3. **Compound table** (`compound_table.pkl`)

   - Format: `pandas.DataFrame`
   - Columns:
     - `compound_id`
     - `smiles` (canonical or standardized SMILES string)

These file paths are specified in the `config` under `data.expression_table`, `data.metadata_table`, and `data.compound_table`.

### 5.2. Configuring thresholds and splits

Within the config JSON:

```json
"data": {
  "expression_table": "path/to/expression_table.pkl",
  "metadata_table": "path/to/metadata_table.pkl",
  "compound_table": "path/to/compound_table.pkl",
  "split_scheme": "random",
  "n_folds": 5,
  "split_seed": 123,
  "up_threshold": 1.0,
  "down_threshold": -1.0
}
```

- `up_threshold` and `down_threshold` are used to derive direction-of-change labels from continuous expression.
- `split_scheme` can be extended to support scaffold splits, cell-held-out splits, etc., depending on your experiments.

---

## 6. Configuration and Training

### 6.1. Experiment configuration

A typical JSON config (like `demo_workspace/configs/demo_config.json`) looks like:

```json
{
  "data": { ... },
  "exposure": {
    "fourier_frequencies": 4,
    "log_eps": 1e-3,
    "include_raw_time_dose": true
  },
  "chem": {
    "train_randomized_smiles": false,
    "randomized_smiles_prob": 0.0,
    "encoder_model_name": "chemberta-small",
    "freeze_encoder": true
  },
  "training": {
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 32,
    "num_epochs": 50,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda",
    "mixed_precision": true,
    "early_stopping_patience": 10,
    "warmup_steps": 0,
    "gradient_clip_norm": 1.0,
    "save_dir": "runs",
    "experiment_name": "my_experiment"
  },
  "loss": {
    "huber_delta": 1.0,
    "listnet_temperature": 1.0,
    "sobolev_first": 0.0,
    "sobolev_second": 0.0,
    "lambda_regression": 1.0,
    "lambda_listnet": 0.0,
    "lambda_sobolev": 0.0,
    "lambda_monotonicity": 0.0,
    "lambda_quantile": 0.0
  },
  "quantile": {
    "enabled": false,
    "quantiles": [0.1, 0.5, 0.9]
  }
}
```

You can use the demo config as a template and adjust the paths and hyperparameters for your dataset.

### 6.2. Launching training manually

Once your config is ready:

```bash
python scripts/train_expo.py --config path/to/config.json
```

The script will:

- Set seeds
- Construct datasets/dataloaders
- Build the ExPO model
- Train with early stopping
- Log metrics to JSONL
- Save checkpoints under `runs/<experiment_name>/`

---

## 7. Calibration & Uncertainty

The `expo.calibration` package provides several utilities that can be used either during or after training:

- **Quantile calibration**
  - `QuantileCalibrator` and `calibrate_groupwise_quantiles` support scaling of quantile outputs to correct under-/over-dispersion, optionally per group (e.g., cell line).
- **Conformal prediction**
  - `MultiOutputConformalRegressor` and `inductive_conformal_interval` support split conformal intervals over single or multiple genes.
- **Temperature scaling**
  - `TemperatureScaler` performs post-hoc calibration of logits for direction-of-change classification.
- **Reliability analysis**
  - `regression_reliability_curve`, `expected_calibration_error`, `root_mean_squared_calibration_error`, `brier_score` provide diagnostics for probabilistic and regression calibration.

These are intended to complement the core ExPO model and are particularly important when interpreting predicted expression levels or derived downstream quantities.

---

## 8. Plotting & Reporting

The repository includes plotting utilities to help visualize and summarize training:

- `scripts/plot_training_curves.py`
  - Reads `metrics.jsonl` for a given run
  - Plots train vs. validation curves (e.g., MAE over epochs)
  - Produces simple histograms of train–validation gaps

- `scripts/plot_uncertainty_curves.py`
  - Given an `.npz` file of predictions, uncertainties, and outcomes, generates:
    - Risk–coverage curves
    - Calibration plots
    - Interval width vs. error diagnostics

- `expo.reporting`
  - `metrics_dict_to_dataframe` and `compare_models_table` for turning nested metrics dictionaries into publication-ready tables
  - `dataframe_to_markdown` and `save_markdown_report` for creating markdown reports summarizing experiment results

These tools are designed to bridge the gap between raw training runs and figure/table generation for manuscripts.

---

## 9. Citation


```text
[Under Review]
```

---

## 10. License

```text
MIT License

Copyright (c) [2025]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 11. Acknowledgements

This implementation builds on ideas from:

- Neural operators and sequence models for structured prediction  
- Chemically informed encoders (e.g., ChemBERTa and related SMILES models)  
- Conformal prediction and calibration methods for regression and classification  

Please see the manuscript and the in-code comments for more detailed references and attributions.

If you have questions, encounter issues, or plan to extend the framework, contributions and discussions are very welcome.
---

## 12. L1000 / Connectivity Map Data

The real experiments described in the accompanying manuscript are based on the **LINCS L1000 / Connectivity Map (CMap)** data produced by the LINCS Center for Transcriptomics at the Broad Institute. In brief, the L1000 assay is a high‑throughput gene expression platform that directly measures 978 “landmark” genes and computationally infers the remaining transcriptome. It has been used to generate more than a million perturbational gene expression profiles across many compounds, genetic perturbations, cell lines, doses, and exposure times.


### 12.1. Primary data sources

All LINCS‑funded L1000 data that populate the CMap resource are deposited in the **NCBI Gene Expression Omnibus (GEO)**. The key accessions are:

- **GSE92742** – Phase I L1000 Connectivity Map perturbational profiles (pilot phase)  
- **GSE70138** – Phase II “production” L1000 Connectivity Map data  
- **GSE92743** – Additional validation / contest datasets (e.g., CMap‑HBS‑LINCS)  

These series contain the full L1000 processing “levels”, from raw Luminex bead intensities (Level 1) up through normalized, replicate‑aggregated signatures (Level 5). For most downstream modeling tasks, including ExPO‑style signature prediction, **Level 5 data** are typically recommended by the data providers. citeturn0search0turn0search1turn0search6


### 12.2. CLUE / CMap web portal

In addition to GEO, the L1000 / CMap data are accessible through the **CLUE platform** (CMap and LINCS Unified Environment):

- CLUE portal: https://clue.io/
- L1000 / CMap data overview: https://clue.io/lincs
- “How can I download the LINCS CMap L1000 data?” (Connectopedia article): https://clue.io/connectopedia/lincs_cmap_data  

The CLUE portal provides higher‑level tools for querying L1000 signatures, running connectivity analyses, and obtaining curated data bundles. It is often the most convenient entry point if you want to explore CMap data interactively before exporting it for use with ExPO. citeturn0search0turn0search1turn0search7turn0search8


### 12.3. Mapping real L1000 data to this codebase

The **synthetic data** created by `scripts/generate_test_data.py` is designed only for testing and demonstration of the pipeline. To run ExPO on real L1000 data, you would:

1. Download the desired subset of L1000 data (e.g., Level 5 signatures) from GEO or CLUE.  
2. Construct three tables to match the expected schema:
   - `expression_table`: rows = perturbational profiles; columns = `profile_id` + gene expression values (e.g., L1000 landmark genes and/or inferred genes).  
   - `metadata_table`: `profile_id`, `compound_id` (or perturbagen ID), `cell_id`, `time`, `dose`, plus any additional covariates you wish to use.  
   - `compound_table`: `compound_id` and `smiles` (or other structural representation) matching the IDs used in the metadata.  
3. Point the `data.expression_table`, `data.metadata_table`, and `data.compound_table` fields in your ExPO config JSON to the corresponding files.  
4. Adjust thresholds (e.g., z‑score cutoffs) and split schemes to reflect the experimental design you are reproducing (random splits, scaffold splits, cell‑held‑out splits, etc.).  

The synthetic generator mirrors this structure so that swapping in real L1000 tables is primarily a matter of **reformatting and configuration**, rather than changing core model code.

