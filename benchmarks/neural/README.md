# Neural model benchmark harness

Benchmarks Chronax's neural forecasters against their Nixtla `neuralforecast`
counterparts on the same protocol, comparing **accuracy** and **CPU speed**.

Wired models: **GRU, PatchTST, KAN, TFT, iTransformer**.

## Prerequisites

- The project environment (the repo's `.venv`) so `import chronax` works.
- For the `neuralforecast` side only, an isolated venv (its PyTorch/Lightning stack
  conflicts with the JAX env), created once:

```bash
bash benchmarks/setup_nf_venv.sh
```

## Run it

From the repo root, with the project environment active:

```bash
# one model, both libraries (Chronax vs neuralforecast)
python benchmarks/neural/run.py --models GRU

# all models, all datasets, both libraries
python benchmarks/neural/run.py

# Chronax only — no neuralforecast venv needed
python benchmarks/neural/run.py --models GRU --libs chronax

# a subset of datasets
python benchmarks/neural/run.py --models PatchTST --datasets AirlinePassengers

# crash-safe: skip rows already written
python benchmarks/neural/run.py --resume
```

- **Models:** `GRU`, `PatchTST`, `KAN`, `TFT`, `iTransformer`
- **Datasets:** `AirlinePassengers`, `DailyFemaleBirths`, `RoomTemperature`

> If `python` isn't the project interpreter on your machine, use `.venv/bin/python`.

## What it measures

- Univariate series; forecast horizon `h=24`, lookback window `input_size=72`.
- **5 seeds** per (model, dataset), identical for both libraries; the first seed is
  a warm-up (JIT / library compile) and is excluded from the timing mean.
- **Accuracy:** MAE (the gate's decision metric) and sMAPE, on the held-out last
  `h` steps.
- **Speed:** after-warmup wall-clock of `fit + predict`; both libraries are pinned
  to the same thread budget for a fair CPU comparison.
- **Accept-gate:** per (model, dataset), PASS if Chronax's mean MAE ≤ NF's **and**
  its after-warmup wall-clock is faster. Printed as a table — a report, not a hard
  failure.

Chronax loops its seeds in one process (JIT cache reuse → meaningful after-warmup
timing); `neuralforecast` runs in `benchmarks/.venv-nf` (one process per seed).

## Output

Per-run CSVs (raw rows + a summary) are written to
`benchmarks/benchmark_results/neural/` (gitignored). Committed reference NF
baselines live under `benchmarks/baselines/<model>/`.

```bash
# (re)capture a model's committed NF baseline (long run; commit the result)
python benchmarks/neural/run.py --refresh-baseline GRU

# compare Chronax against the committed baseline without re-running NF
python benchmarks/neural/run.py --check-committed --models GRU
```

## Configuration

Everything is declared in `benchmarks/neural/config.yaml`: horizon, lookback,
seeds, thread budget, the datasets, and each model's Chronax / NF hyperparameters.

## Adding a model

A model qualifies if it is a Chronax `BaseForecaster` accepting `h`, `input_size`,
and `random_seed`, **and** has a faithful `neuralforecast` counterpart. To add it:

1. Add one entry to `_NF_NAMES` in `benchmarks/neural/registry.py`.
2. Add a `models:` block in `config.yaml` with its `chronax_params` / `nf_params`.

## Tests

```bash
python -m pytest tests/neural/ -q
```

Offline and deterministic; the `neuralforecast` smoke test skips automatically when
`benchmarks/.venv-nf` is absent.
