# Neural benchmark harness

How to run the neural benchmarks (Chronax vs Nixtla `neuralforecast`).

## Setup (one-time)

The `neuralforecast` side runs in an isolated venv (its PyTorch/Lightning stack
conflicts with the JAX env):

```bash
# Unix / Git Bash / WSL
bash benchmarks/setup_nf_venv.sh
```

```powershell
# Windows PowerShell
python -m venv benchmarks\.venv-nf
.\benchmarks\.venv-nf\Scripts\pip.exe install -r benchmarks\requirements-nf.txt
```

The harness auto-detects `Scripts\python.exe` (Windows) or `bin/python` (Unix).

## Run

From the repo root, with the project environment active:

```bash
# everything (all models, all datasets, both libraries)
python benchmarks/neural/run.py

# one model
python benchmarks/neural/run.py --models GRU

# one dataset
python benchmarks/neural/run.py --datasets AirlinePassengers

# Chronax only — no neuralforecast venv needed
python benchmarks/neural/run.py --libs chronax

# resume a crashed run
python benchmarks/neural/run.py --resume
```

Models are auto-discovered from `chronax.models` — any neural forecaster in the
library's public list is benchmarked automatically, no wiring needed. Datasets and
run parameters (horizon, lookback, seeds, threads) are defined in
`benchmarks/neural/config.yaml`, along with optional per-model overrides.

> If `python` isn't the project interpreter, use `.venv/bin/python`.

## Dataset kinds

Each entry under `datasets:` has a `kind` (default `univariate`):

| kind | Config fields | Evaluation |
|------|---------------|------------|
| `univariate` | `y_col` | 1-D target (legacy Airline / Births / RoomTemp) |
| `multivariate` | `y_cols` (2–4) | joint targets; Chronax fits each channel independently; NF uses one `unique_id` per channel and `n_series=N` when required |
| `covariate` | `y_col` + `hist_exog_cols` | target + exogenous; Chronax passes `X`/`futr_exog` only when `uses_exog=True`; NF sets `hist_exog_list` / `futr_exog_list` when the model accepts them |

Optional per-dataset `h` / `input_size` override the experiment defaults (used for short GA4 series).

Wide CSVs live under `benchmarks/benchmark_datasets/` in the same layout as the
univariate files (`Datetime` + numeric columns). Regenerate the TempusBench-derived
set with:

```bash
python benchmarks/neural/import_tempus_datasets.py
```

### TempusBench-derived datasets (non-Kaggle, ≤4 features)

| Name | Kind | Source |
|------|------|--------|
| `GA4EcommerceKPI` | multivariate | Google Analytics BigQuery sample |
| `WeeklyCushingUS` | multivariate | EIA Cushing / Weekly Petroleum |
| `SplitSmartACEnergy` | multivariate | SplitSmart AC (data.gov); 4 dense sensors |
| `GA4RevenueTraffic` | covariate | GA4 revenue + traffic mix |
| `GEFCom2014PV` | covariate | GEFCom2014 PV + weather |
| `CapitalBikeshare` | covariate | UCI Capital Bikeshare |

## Output

Results go to `benchmarks/benchmark_results/neural/` (gitignored) — per-run CSVs
plus a printed accuracy/speed comparison. Committed reference baselines live under
`benchmarks/baselines/`:

```bash
# (re)capture a model's baseline
python benchmarks/neural/run.py --refresh-baseline GRU

# compare against the committed baseline without re-running neuralforecast
python benchmarks/neural/run.py --check-committed --models GRU
```

## Tests

```bash
python -m pytest tests/neural/ -q
```
