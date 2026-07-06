# Nixtla `neuralforecast` baselines (per model)

Reference numbers the Chronax neural models must clear on the same protocol.
One directory per model: `GRU/`, `PatchTST/`, `KAN/`.

## Files (per model dir)

| File | Purpose |
|---|---|
| `protocol.json` | Exact protocol (h, input_size, seeds, warmup_seeds, threads), the config NF params, the fully-resolved NF architecture (`resolved_model_args`), and captured env `versions` (python/torch/neuralforecast/platform). |
| `baseline_nixtla_summary.csv` | Mean ± std over 5 seeds (MAE, sMAPE, wall-clock) + after-warmup wall-clock. |
| `baseline_nixtla_raw.csv` | Per-seed raw rows. |

## Reproducing

From a clean clone (activate the project venv first):

```bash
# 1. Create the isolated Nixtla venv (one-time).
bash benchmarks/setup_nf_venv.sh

# 2a. Full comparison (both libraries) + accept-gate:
python benchmarks/neural/run.py

# 2b. Recapture a model's NF baseline under the pinned-thread regime:
python benchmarks/neural/run.py --refresh-baseline GRU
```

Hardware caveats: baselines are captured CPU-only under a pinned thread budget
(`experiment.threads` in `benchmarks/neural/config.yaml`); the exact thread
count, resolved NF args, and captured library versions are recorded in each
`protocol.json`. Different hardware shifts wall-clock; the comparison is valid
as "same machine, same nominal thread budget" because both libraries run pinned
on the same host.
