"""Neural benchmark worker: one isolated process per {model, dataset, library}.

Chronax loops seeds in-process (JIT cache reused for seeds 2..N -> after-warmup
timing is meaningful). NF spawns a fresh .venv-nf process per seed (clean per-seed
RNG; matches how baselines are captured). Both libraries are pinned to
experiment.threads. One `RESULT_JSON:::{...}` line is streamed per seed.

IMPORTANT: this module must NOT import jax/chronax at top level — threads are
pinned via env vars that must be set BEFORE JAX initializes (see main()).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"


def build_thread_env(threads: int) -> dict[str, str]:
    """Env vars pinning XLA (Chronax) CPU parallelism to `threads`.

    NOTE: `intra_op_parallelism_threads` is a TensorFlow session option and may
    be ignored by XLA; the effective CPU pin rests on OMP_NUM_THREADS +
    xla_cpu_multi_thread_eigen. The actual Chronax thread pin is validated
    empirically during the Task 14 calibration, not assumed here.
    """
    threads = int(threads)
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    multithread = "false" if threads == 1 else "true"
    return {
        "OMP_NUM_THREADS": str(threads),
        "XLA_FLAGS": (
            f"--xla_cpu_multi_thread_eigen={multithread} "
            f"intra_op_parallelism_threads={threads}"
        ),
    }


def _nf_kwargs_str(params: dict) -> str:
    """Render NF constructor kwargs from config nf_params. loss -> MAE()."""
    parts: list[str] = []
    for k, v in params.items():
        if k == "loss":
            if v != "MAE":
                raise ValueError(f"only loss=MAE is supported for NF, got {v!r}")
            parts.append("loss=MAE()")
        elif isinstance(v, str):
            parts.append(f"{k}={v!r}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def nf_subprocess_code(nf_name: str, spec: dict[str, str], h: int, input_size: int, params: dict, seed: int, threads: int) -> str:
    """Build the `python -c` source run inside .venv-nf for one NF seed.

    torch.set_num_threads pins CPU threads; the neuralforecast import is BEFORE
    t0 so the timer measures only fit+predict (fair timing, spec §1).
    """
    kwargs = _nf_kwargs_str(params)
    return f'''
import json, time, numpy as np, pandas as pd
import torch
torch.set_num_threads({int(threads)})
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import {nf_name}
df = pd.read_csv({spec['path']!r})
df = df.rename(columns={{ {spec['ds_col']!r}: 'ds', {spec['y_col']!r}: 'y' }})
df['ds'] = pd.to_datetime(df['ds']); df['unique_id'] = {spec['name']!r}
df = df[['unique_id','ds','y']].sort_values('ds').reset_index(drop=True)
train, test = df.iloc[:-{h}], df.iloc[-{h}:]
y_true = test['y'].to_numpy()
m = {nf_name}(h={h}, input_size={input_size}, {kwargs},
    random_seed={seed}, accelerator='cpu', enable_progress_bar=False,
    logger=False, enable_model_summary=False, enable_checkpointing=False)
nf = NeuralForecast(models=[m], freq={spec['freq']!r})
t0 = time.perf_counter(); nf.fit(df=train); fcst = nf.predict(); t = time.perf_counter() - t0
y_hat = fcst[{nf_name!r}].to_numpy()
mae = float(np.mean(np.abs(y_true - y_hat)))
denom = (np.abs(y_true) + np.abs(y_hat)) / 2
smape = float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_hat) / denom)) * 100)
print(json.dumps({{'mae': mae, 'smape': smape, 'wallclock': t}}))
'''

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(REPO))
from benchmarks.neural import metrics, registry  # noqa: E402
from benchmarks.neural.run import load_config  # noqa: E402


def _dataset_spec(cfg: dict, name: str) -> dict:
    for d in cfg["datasets"]:
        if d["name"] == name:
            spec = dict(d)
            p = Path(spec["path"])
            spec["path"] = str(p if p.is_absolute() else (REPO / p).resolve())
            return spec
    raise KeyError(f"unknown dataset {name!r}")


def _model_cfg(cfg: dict, name: str) -> dict:
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"unknown model {name!r}")


def load_dataset_y(spec: dict) -> np.ndarray:
    import pandas as pd
    return pd.read_csv(spec["path"])[spec["y_col"]].to_numpy(dtype=np.float32)


def run_chronax_seed(cls: type, y_train: np.ndarray, y_test: np.ndarray, h: int, input_size: int, chronax_params: dict, seed: int) -> dict:
    """Fit+predict one Chronax seed; return metric row with an `error` field."""
    import jax.numpy as jnp
    row = {"mae": None, "smape": None, "wallclock_s": None, "error": ""}
    try:
        model = cls(h=h, input_size=input_size, random_seed=seed, **chronax_params)
        t0 = time.perf_counter()
        model.fit(jnp.asarray(y_train))
        pred = np.asarray(model.predict(h=h)["mean"])
        elapsed = time.perf_counter() - t0
        row.update(mae=metrics.mae(y_test, pred),
                   smape=metrics.smape(y_test, pred), wallclock_s=elapsed)
    except Exception as e:  # noqa: BLE001
        # Documented broad catch (spec §10): a diverged Chronax fit raises a bare
        # RuntimeError (gru_model.py); a typed exception is a chronax/ change out
        # of scope. Record the message and continue so one seed can't kill a run.
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def emit_row(library: str, dataset: str, model: str, seed: int, iter_idx: int, warmup_seeds: int, row: dict) -> None:
    out = {
        "library": library, "dataset": dataset, "model": model, "seed": seed,
        "iter_idx": iter_idx, "is_warmup": iter_idx < warmup_seeds,
        "mae": row["mae"], "smape": row["smape"], "wallclock_s": row["wallclock_s"],
        "error": row.get("error", ""),
    }
    print(f"RESULT_JSON:::{json.dumps(out)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Neural benchmark worker (one library).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--library", required=True, choices=["chronax", "nixtla"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Override the seed list (used by --resume to run remaining seeds).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    h, input_size = exp["h"], exp["input_size"]
    seeds = args.seeds if args.seeds is not None else exp["seeds"]
    warmup, threads = exp["warmup_seeds"], exp["threads"]
    spec = _dataset_spec(cfg, args.dataset)
    model_cfg = _model_cfg(cfg, args.model)

    # Pin threads BEFORE importing jax/chronax (see module docstring).
    os.environ.update(build_thread_env(threads))

    if args.library == "chronax":
        cls = registry.resolve_chronax(args.model)
        y = load_dataset_y(spec)
        y_train, y_test = y[:-h], y[-h:]
        for i, seed in enumerate(seeds):
            row = run_chronax_seed(cls, y_train, y_test, h, input_size,
                                   model_cfg["chronax_params"], seed)
            emit_row(args.library, args.dataset, args.model, seed, i, warmup, row)
    else:
        nf_name = registry.nf_model_name(args.model)
        for i, seed in enumerate(seeds):
            row = run_nixtla_seed(nf_name, spec, h, input_size,
                                  model_cfg["nf_params"], seed, threads)
            emit_row(args.library, args.dataset, args.model, seed, i, warmup, row)


if __name__ == "__main__":
    main()
