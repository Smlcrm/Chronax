"""GRU comparison benchmark: Chronax vs Nixtla neuralforecast.

Run with:
    .venv/bin/python benchmarks/gru_benchmark.py

The Nixtla side is invoked as a subprocess into ``benchmarks/.venv-nf`` so the
main Chronax env stays clean of torch + Lightning.

Reports both total wall-clock (with JIT/Lightning warmup) and after-warmup
wall-clock (first fit per dataset excluded). The acceptance comparison uses
after-warmup time. Reference baselines and the protocol live under
``benchmarks/baselines/``; see that directory's README for the rule that
``benchmarks/gru_benchmark.py`` must clear.

Outputs:
    benchmarks/benchmark_results/gru_<timestamp>.csv          (raw per-run rows)
    benchmarks/benchmark_results/gru_<timestamp>_summary.csv  (mean/std summary)
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

from chronax.models import GRU

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO / "benchmarks" / "benchmark_datasets"
RESULTS_DIR = REPO / "benchmarks" / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"

H = 24
INPUT_SIZE = 72
SEEDS = [42, 43, 44, 45, 46]

DATASETS = [
    {
        "name": "AirlinePassengers",
        "path": str(DATASETS_DIR / "airline-passengers.csv"),
        "ds_col": "Month", "y_col": "Passengers", "freq": "MS",
    },
    {
        "name": "DailyFemaleBirths",
        "path": str(DATASETS_DIR / "daily-total-female-births.csv"),
        "ds_col": "Date", "y_col": "Births", "freq": "D",
    },
    {
        "name": "RoomTemperature",
        "path": str(DATASETS_DIR / "room temperature data.csv"),
        "ds_col": "Datetime", "y_col": "Hourly_Temp", "freq": "h",
    },
]


def load_y(spec: dict) -> np.ndarray:
    return pd.read_csv(spec["path"])[spec["y_col"]].to_numpy(dtype=np.float32)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def smape(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.abs(a) + np.abs(b)) / 2.0
    return float(np.mean(np.where(denom == 0.0, 0.0, np.abs(a - b) / denom)) * 100.0)


def run_chronax(spec: dict, seed: int) -> tuple[float, float, float]:
    y = load_y(spec)
    y_train, y_test = y[:-H], y[-H:]
    model = GRU(h=H, input_size=INPUT_SIZE, random_seed=seed)
    t0 = time.perf_counter()
    model.fit(jnp.asarray(y_train))
    pred = np.asarray(model.predict(h=H)["mean"])
    elapsed = time.perf_counter() - t0
    return mae(y_test, pred), smape(y_test, pred), elapsed


def run_nixtla(spec: dict, seed: int) -> tuple[float, float, float]:
    """Invoke neuralforecast in the isolated venv via subprocess."""
    code = f"""
import json, time, numpy as np, pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import GRU
df = pd.read_csv({spec['path']!r})
df = df.rename(columns={{ {spec['ds_col']!r}: 'ds', {spec['y_col']!r}: 'y' }})
df['ds'] = pd.to_datetime(df['ds']); df['unique_id'] = {spec['name']!r}
df = df[['unique_id','ds','y']].sort_values('ds').reset_index(drop=True)
train, test = df.iloc[:-{H}], df.iloc[-{H}:]
y_true = test['y'].to_numpy()
m = GRU(h={H}, input_size={INPUT_SIZE}, max_steps=1000, learning_rate=1e-3,
        scaler_type='robust', random_seed={seed}, loss=MAE(),
        accelerator='cpu', enable_progress_bar=False, logger=False,
        enable_model_summary=False, enable_checkpointing=False)
nf = NeuralForecast(models=[m], freq={spec['freq']!r})
t0 = time.perf_counter(); nf.fit(df=train); fcst = nf.predict(); t = time.perf_counter() - t0
y_hat = fcst['GRU'].to_numpy()
mae = float(np.mean(np.abs(y_true - y_hat)))
denom = (np.abs(y_true) + np.abs(y_hat)) / 2
smape = float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_hat) / denom)) * 100)
print(json.dumps({{'mae': mae, 'smape': smape, 'wallclock': t}}))
"""
    out = subprocess.run(
        [str(NF_VENV_PY), "-c", code], capture_output=True, text=True, check=True
    )
    last_json_line = out.stdout.strip().splitlines()[-1]
    d = json.loads(last_json_line)
    return d["mae"], d["smape"], d["wallclock"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="+", default=["chronax", "nixtla"],
                    choices=["chronax", "nixtla"])
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Subset of dataset names to run.")
    args = ap.parse_args()

    datasets = (
        DATASETS if args.datasets is None
        else [d for d in DATASETS if d["name"] in args.datasets]
    )

    rows: list[dict] = []
    for spec in datasets:
        print(f"\n=== {spec['name']} ===")
        for lib in args.libs:
            for i, seed in enumerate(SEEDS):
                fn = run_chronax if lib == "chronax" else run_nixtla
                m, s, t = fn(spec, seed)
                rows.append({
                    "library": lib, "dataset": spec["name"], "seed": seed,
                    "iter_idx": i, "is_warmup": i == 0,
                    "mae": m, "smape": s, "wallclock_s": t,
                })
                print(f"  {lib:8s} seed={seed} mae={m:.4f} smape={s:.4f} t={t:.1f}s")

    df = pd.DataFrame(rows)
    summary_total = df.groupby(["library", "dataset"]).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        smape_mean=("smape", "mean"), smape_std=("smape", "std"),
        wallclock_mean=("wallclock_s", "mean"),
        wallclock_std=("wallclock_s", "std"),
    ).round(4)
    after = df[~df["is_warmup"]].groupby(["library", "dataset"]).agg(
        wallclock_mean_after=("wallclock_s", "mean"),
        wallclock_std_after=("wallclock_s", "std"),
    ).round(4)

    print("\n=== Total wall-clock (5 seeds) ===\n", summary_total.to_string())
    print("\n=== After-warmup wall-clock (4 seeds, used for acceptance) ===\n",
          after.to_string())

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df.to_csv(RESULTS_DIR / f"gru_{ts}.csv", index=False)
    summary_total.join(after).to_csv(RESULTS_DIR / f"gru_{ts}_summary.csv")
    print(f"\nWrote: gru_{ts}.csv  gru_{ts}_summary.csv")


if __name__ == "__main__":
    main()
