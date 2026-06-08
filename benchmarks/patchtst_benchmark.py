"""PatchTST comparison benchmark: Chronax vs Nixtla neuralforecast.

Run with (long job; rows are written incrementally so it is crash-safe):
    .venv/bin/python benchmarks/patchtst_benchmark.py --resume

The Nixtla side runs in ``benchmarks/.venv-nf`` (subprocess) at PatchTST's native
defaults — RevIN + ``scaler_type='identity'``, ``max_steps=5000``,
``windows_batch_size=1024`` — so the comparison is faithful, not confounded by a
weakened reference. Reports total wall-clock (with JIT/Lightning warmup) and
after-warmup wall-clock (first seed per dataset excluded); acceptance uses
after-warmup time. Each (library, dataset, seed) row is appended to the output
CSV as it finishes; ``--resume`` skips rows already present so an interrupted run
continues where it left off.

Outputs (stable names so ``--resume`` works):
    benchmarks/benchmark_results/patchtst_results.csv          (raw per-run rows)
    benchmarks/benchmark_results/patchtst_results_summary.csv  (mean/std summary)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from chronax.models import PatchTST

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "benchmarks" / "benchmark_datasets"
RESULTS_DIR = REPO / "benchmarks" / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"

H = 24
INPUT_SIZE = 72
MAX_STEPS = 5000
SEEDS = [42, 43, 44, 45, 46]

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


def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0
    return float(np.mean(np.where(denom == 0.0, 0.0, np.abs(a - b) / denom)) * 100.0)


def run_chronax(spec, seed):
    y = load_y(spec)
    y_train, y_test = y[:-H], y[-H:]
    model = PatchTST(h=H, input_size=INPUT_SIZE, max_steps=MAX_STEPS, random_seed=seed)
    t0 = time.perf_counter()
    model.fit(jnp.asarray(y_train))
    pred = np.asarray(model.predict(h=H)["mean"])
    elapsed = time.perf_counter() - t0
    return mae(y_test, pred), smape(y_test, pred), elapsed


def run_nixtla(spec, seed):
    code = f"""
import json, time, numpy as np, pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import PatchTST
df = pd.read_csv({spec['path']!r})
df = df.rename(columns={{ {spec['ds_col']!r}: 'ds', {spec['y_col']!r}: 'y' }})
df['ds'] = pd.to_datetime(df['ds']); df['unique_id'] = {spec['name']!r}
df = df[['unique_id','ds','y']].sort_values('ds').reset_index(drop=True)
train, test = df.iloc[:-{H}], df.iloc[-{H}:]
y_true = test['y'].to_numpy()
m = PatchTST(h={H}, input_size={INPUT_SIZE}, max_steps={MAX_STEPS},
             learning_rate=1e-4, random_seed={seed}, loss=MAE(),
             accelerator='cpu', enable_progress_bar=False, logger=False,
             enable_model_summary=False, enable_checkpointing=False)
nf = NeuralForecast(models=[m], freq={spec['freq']!r})
t0 = time.perf_counter(); nf.fit(df=train); fcst = nf.predict(); t = time.perf_counter() - t0
y_hat = fcst['PatchTST'].to_numpy()
mae = float(np.mean(np.abs(y_true - y_hat)))
denom = (np.abs(y_true) + np.abs(y_hat)) / 2
smape = float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_hat) / denom)) * 100)
print(json.dumps({{'mae': mae, 'smape': smape, 'wallclock': t}}))
"""
    out = subprocess.run([str(NF_VENV_PY), "-c", code], capture_output=True, text=True, check=True)
    d = json.loads(out.stdout.strip().splitlines()[-1])
    return d["mae"], d["smape"], d["wallclock"]


_FIELDS = ["library", "dataset", "seed", "iter_idx", "is_warmup", "mae", "smape", "wallclock_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="+", default=["chronax", "nixtla"], choices=["chronax", "nixtla"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "patchtst_results.csv"),
                    help="Stable CSV path; each (lib,dataset,seed) row is appended as it finishes.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (library,dataset,seed) rows already present in --out.")
    args = ap.parse_args()
    out_path = Path(args.out)
    datasets = DATASETS if args.datasets is None else [d for d in DATASETS if d["name"] in args.datasets]

    done = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        done = {(r.library, r.dataset, int(r.seed)) for r in prev.itertuples()}
        print(f"resume: {len(done)} rows already done in {out_path.name}", flush=True)

    write_header = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()
        for spec in datasets:
            print(f"\n=== {spec['name']} ===", flush=True)
            for lib in args.libs:
                for i, seed in enumerate(SEEDS):
                    if (lib, spec["name"], seed) in done:
                        print(f"  {lib:8s} seed={seed} (skip, already done)", flush=True)
                        continue
                    fn = run_chronax if lib == "chronax" else run_nixtla
                    m, s, t = fn(spec, seed)
                    writer.writerow({"library": lib, "dataset": spec["name"], "seed": seed,
                                     "iter_idx": i, "is_warmup": i == 0,
                                     "mae": m, "smape": s, "wallclock_s": t})
                    f.flush()  # crash-safe: row is on disk before the next (long) fit
                    print(f"  {lib:8s} seed={seed} mae={m:.4f} smape={s:.4f} t={t:.1f}s", flush=True)

    if not out_path.exists():
        return
    df = pd.read_csv(out_path)
    summary_total = df.groupby(["library", "dataset"]).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        smape_mean=("smape", "mean"), smape_std=("smape", "std"),
        wallclock_mean=("wallclock_s", "mean"), wallclock_std=("wallclock_s", "std"),
    ).round(4)
    after = df[~df["is_warmup"]].groupby(["library", "dataset"]).agg(
        wallclock_mean_after=("wallclock_s", "mean"), wallclock_std_after=("wallclock_s", "std"),
    ).round(4)
    print("\n=== Total wall-clock ===\n", summary_total.to_string())
    print("\n=== After-warmup wall-clock (used for acceptance) ===\n", after.to_string())
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary_total.join(after).to_csv(summary_path)
    print(f"\nWrote rows -> {out_path.name}, summary -> {summary_path.name}", flush=True)


if __name__ == "__main__":
    main()
