# Nixtla `neuralforecast.GRU` baselines

Captured **2026-05-05**.

These are the reference numbers that `benchmarks/gru_benchmark.py` must clear
on the same protocol — see `benchmarks/baselines/baseline_nixtla_gru_protocol.json`
for exact hyperparameters.

## Files

| File | Purpose |
|---|---|
| `baseline_nixtla_gru_protocol.json` | The exact protocol (h, input_size, seeds, model args). |
| `baseline_nixtla_gru_summary.csv` | Mean ± std over 5 seeds (MAE, sMAPE, wall-clock). |
| `baseline_nixtla_gru_raw.csv` | Per-seed raw rows. |

## Reproducing

From a clean clone:

```bash
# 1. Create the isolated Nixtla venv (one-time).
bash benchmarks/setup_nf_venv.sh

# 2. Run the comparison benchmark (writes to benchmarks/benchmark_results/).
.venv/bin/python benchmarks/gru_benchmark.py
```

Hardware caveats: numbers in this directory were captured on Apple M3
(macOS 25.4), CPU-only, default thread settings. Different hardware will
shift wall-clock; relative comparisons via `gru_benchmark.py` remain valid
because both libraries run on the same machine.
