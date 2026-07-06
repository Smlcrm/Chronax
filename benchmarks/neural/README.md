# Neural benchmark harness

How to run the neural benchmarks (Chronax vs Nixtla `neuralforecast`).

## Setup (one-time)

The `neuralforecast` side runs in an isolated venv (its PyTorch/Lightning stack
conflicts with the JAX env):

```bash
bash benchmarks/setup_nf_venv.sh
```

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

The models, datasets, and all run parameters (horizon, lookback, seeds, threads)
are defined in `benchmarks/neural/config.yaml`.

> If `python` isn't the project interpreter, use `.venv/bin/python`.

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
