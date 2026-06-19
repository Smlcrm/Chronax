"""iTransformer comparison benchmark: Chronax vs Nixtla neuralforecast.

Run with:
    .venv/bin/python benchmarks/itransformer_benchmark.py

The Nixtla side runs in ``benchmarks/.venv-nf`` (subprocess) at the SAME
hyperparameters as the Chronax side (``n_series=1``, ``use_norm=True``,
matching ``hidden_size``/``n_heads``/``e_layers``/``d_ff``/``max_steps``/
``learning_rate``/``windows_batch_size``), so the comparison is faithful — any
difference comes from the engine (JAX/JIT vs PyTorch-Lightning), not a weakened
reference.

Reports MAE, RMSE, MAPE, SMAPE and wall-clock. Two wall-clock views: total
(includes JIT / Lightning warmup) and after-warmup (first fit per dataset
excluded). Acceptance uses after-warmup time (the "warm start" the project
optimizes for) and the accuracy metrics.

Outputs (NOT committed — benchmark result CSVs stay local for review with Denny):
    benchmarks/benchmark_results/itransformer_<timestamp>.csv          (raw per-run rows)
    benchmarks/benchmark_results/itransformer_<timestamp>_summary.csv  (mean/std summary)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from chronax.models import iTransformer

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "benchmarks" / "benchmark_datasets"
RESULTS_DIR = REPO / "benchmarks" / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"

# Forecast setup.
H = 24
INPUT_SIZE = 72
SEEDS = [42, 43, 44, 45, 46]

# Model hyperparameters — identical on both sides for a faithful comparison.
# Defaults follow neuralforecast.iTransformer.
HIDDEN_SIZE = 128
N_HEADS = 8
E_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
USE_NORM = True
MAX_STEPS = 1000
LEARNING_RATE = 1e-3
WINDOWS_BATCH_SIZE = 32

DATASETS = [
    {"name": "AirlinePassengers", "path": str(DATASETS_DIR / "airline-passengers.csv"),
     "ds_col": "Month", "y_col": "Passengers", "freq": "MS"},
    {"name": "DailyFemaleBirths", "path": str(DATASETS_DIR / "daily-total-female-births.csv"),
     "ds_col": "Date", "y_col": "Births", "freq": "D"},
    {"name": "RoomTemperature", "path": str(DATASETS_DIR / "room temperature data.csv"),
     "ds_col": "Datetime", "y_col": "Hourly_Temp", "freq": "h"},
]


def load_y(spec):
    return pd.read_csv(spec["path"])[spec["y_col"]].to_numpy(dtype=np.float32)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a, b):
    eps = 1e-10
    return float(np.mean(np.abs((a - b) / (a + eps))) * 100.0)


def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0
    return float(np.mean(np.where(denom == 0.0, 0.0, np.abs(a - b) / denom)) * 100.0)


def _metrics(y_true, y_hat):
    return {"mae": mae(y_true, y_hat), "rmse": rmse(y_true, y_hat),
            "mape": mape(y_true, y_hat), "smape": smape(y_true, y_hat)}


def run_chronax(spec, seed):
    y = load_y(spec)
    y_train, y_test = y[:-H], y[-H:]
    model = iTransformer(
        h=H, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, n_heads=N_HEADS,
        e_layers=E_LAYERS, d_ff=D_FF, dropout=DROPOUT, use_norm=USE_NORM,
        max_steps=MAX_STEPS, learning_rate=LEARNING_RATE,
        windows_batch_size=WINDOWS_BATCH_SIZE, random_seed=seed,
    )
    t0 = time.perf_counter()
    model.fit(jnp.asarray(y_train))
    pred = np.asarray(model.predict(h=H)["mean"])
    elapsed = time.perf_counter() - t0
    m = _metrics(y_test, pred)
    m["wallclock"] = elapsed
    return m


def run_nixtla(spec, seed):
    code = f"""
import json, time, numpy as np, pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import iTransformer
df = pd.read_csv({spec['path']!r})
df = df.rename(columns={{ {spec['ds_col']!r}: 'ds', {spec['y_col']!r}: 'y' }})
df['ds'] = pd.to_datetime(df['ds']); df['unique_id'] = {spec['name']!r}
df = df[['unique_id','ds','y']].sort_values('ds').reset_index(drop=True)
train, test = df.iloc[:-{H}], df.iloc[-{H}:]
y_true = test['y'].to_numpy()
m = iTransformer(h={H}, input_size={INPUT_SIZE}, n_series=1,
                 hidden_size={HIDDEN_SIZE}, n_heads={N_HEADS}, e_layers={E_LAYERS},
                 d_ff={D_FF}, dropout={DROPOUT}, use_norm={USE_NORM},
                 max_steps={MAX_STEPS}, learning_rate={LEARNING_RATE},
                 windows_batch_size={WINDOWS_BATCH_SIZE}, random_seed={seed}, loss=MAE(),
                 accelerator='cpu', enable_progress_bar=False, logger=False,
                 enable_model_summary=False, enable_checkpointing=False)
nf = NeuralForecast(models=[m], freq={spec['freq']!r})
t0 = time.perf_counter(); nf.fit(df=train); fcst = nf.predict(); t = time.perf_counter() - t0
y_hat = fcst['iTransformer'].to_numpy()
eps = 1e-10
mae = float(np.mean(np.abs(y_true - y_hat)))
rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
mape = float(np.mean(np.abs((y_true - y_hat) / (y_true + eps))) * 100)
denom = (np.abs(y_true) + np.abs(y_hat)) / 2
smape = float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_hat) / denom)) * 100)
print(json.dumps({{'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape, 'wallclock': t}}))
"""
    out = subprocess.run([str(NF_VENV_PY), "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    global INPUT_SIZE, MAX_STEPS, HIDDEN_SIZE, D_FF, LEARNING_RATE, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="+", default=["chronax", "nixtla"], choices=["chronax", "nixtla"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--input-size", type=int, default=INPUT_SIZE)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    ap.add_argument("--d-ff", type=int, default=D_FF)
    ap.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    # Both sides read these module globals, so overriding here keeps the comparison matched.
    INPUT_SIZE, MAX_STEPS, HIDDEN_SIZE = args.input_size, args.max_steps, args.hidden_size
    D_FF, LEARNING_RATE, SEEDS = args.d_ff, args.learning_rate, args.seeds
    datasets = DATASETS if args.datasets is None else [d for d in DATASETS if d["name"] in args.datasets]
    print(f"config: input_size={INPUT_SIZE} max_steps={MAX_STEPS} hidden_size={HIDDEN_SIZE} "
          f"d_ff={D_FF} lr={LEARNING_RATE} seeds={SEEDS}")

    if "nixtla" in args.libs and not NF_VENV_PY.exists():
        raise SystemExit(
            f"Nixtla venv not found at {NF_VENV_PY}. Create it with "
            "`bash benchmarks/setup_nf_venv.sh`, or run with `--libs chronax`."
        )

    rows = []
    for spec in datasets:
        print(f"\n=== {spec['name']} ===")
        for lib in args.libs:
            for i, seed in enumerate(SEEDS):
                fn = run_chronax if lib == "chronax" else run_nixtla
                r = fn(spec, seed)
                rows.append({"library": lib, "dataset": spec["name"], "seed": seed,
                             "iter_idx": i, "is_warmup": i == 0,
                             "input_size": INPUT_SIZE, "max_steps": MAX_STEPS, **r,
                             "wallclock_s": r["wallclock"]})
                print(f"  {lib:8s} seed={seed} mae={r['mae']:.4f} rmse={r['rmse']:.4f} "
                      f"mape={r['mape']:.3f} t={r['wallclock']:.1f}s")

    df = pd.DataFrame(rows)
    summary_total = df.groupby(["library", "dataset"]).agg(
        mae_mean=("mae", "mean"), rmse_mean=("rmse", "mean"), mape_mean=("mape", "mean"),
        smape_mean=("smape", "mean"), wallclock_mean=("wallclock_s", "mean"),
        wallclock_std=("wallclock_s", "std"),
    ).round(4)
    after = df[~df["is_warmup"]].groupby(["library", "dataset"]).agg(
        wallclock_mean_after=("wallclock_s", "mean"), wallclock_std_after=("wallclock_s", "std"),
    ).round(4)
    print("\n=== Accuracy + total wall-clock (5 seeds) ===\n", summary_total.to_string())
    print("\n=== After-warmup wall-clock (4 seeds, used for acceptance) ===\n", after.to_string())

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df.to_csv(RESULTS_DIR / f"itransformer_{ts}.csv", index=False)
    summary_total.join(after).to_csv(RESULTS_DIR / f"itransformer_{ts}_summary.csv")
    print(f"\nWrote: itransformer_{ts}.csv  itransformer_{ts}_summary.csv")


if __name__ == "__main__":
    main()
