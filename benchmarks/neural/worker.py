"""Neural benchmark worker: one isolated process per {model, dataset, library}.

Chronax loops seeds in-process (JIT cache reused for seeds 2..N -> after-warmup
timing is meaningful). NF spawns a fresh .venv-nf process per seed (clean per-seed
RNG; matches how baselines are captured). Both libraries are pinned to
experiment.threads. One `RESULT_JSON:::{...}` line is streamed per seed.

Supports dataset kinds: univariate | multivariate | covariate (see config.yaml).

IMPORTANT: this module must NOT import jax/chronax at top level — threads are
pinned via env vars that must be set BEFORE JAX initializes (see main()).
"""
from __future__ import annotations

import sys
import inspect
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _resolve_nf_venv_py(repo_root: Path) -> Path:
    """Windows: Scripts/python.exe; Unix: bin/python."""
    base = Path(repo_root) / "benchmarks" / ".venv-nf"
    candidates = (
        base / "Scripts" / "python.exe",
        base / "bin" / "python",
        base / "bin" / "python3",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if sys.platform.startswith("win") else candidates[1]


NF_VENV_PY = _resolve_nf_venv_py(REPO)


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


def _nf_extra_dict(params: dict) -> str:
    """Render non-loss nf_params as a Python dict literal for the subprocess.

    Loss is fixed to MAE in the constructor template (only loss=MAE is supported).
    n_series / hist_exog_list / futr_exog_list are injected at runtime from the
    dataset kind (see nf_subprocess_code), so they never need to appear here.
    """
    skip = {"loss", "n_series", "hist_exog_list", "futr_exog_list"}
    items = []
    for k, v in params.items():
        if k == "loss":
            if v != "MAE":
                raise ValueError(f"only loss=MAE is supported for NF, got {v!r}")
            continue
        if k in skip:
            continue
        items.append(f"{k!r}: {v!r}")
    return "{" + ", ".join(items) + "}"


def dataset_kind(spec: dict) -> str:
    return str(spec.get("kind", "univariate"))


def resolve_horizon(spec: dict, exp: dict) -> tuple[int, int]:
    """Per-dataset h / input_size override experiment defaults when present."""
    h = int(spec.get("h", exp["h"]))
    input_size = int(spec.get("input_size", exp["input_size"]))
    return h, input_size


def load_dataset(spec: dict) -> dict:
    """Load a wide Chronax-style CSV according to dataset kind.

    Returns a dict with keys:
      kind: str
      y: float32 array — shape (T,) univariate/covariate or (T, N) multivariate
      X: float32 array | None — shape (T, F) for covariate, else None
      y_cols: list[str] | None — multivariate channel names
      hist_exog_cols: list[str] | None — covariate exog names
    """
    import pandas as pd

    kind = dataset_kind(spec)
    df = pd.read_csv(spec["path"])
    if kind == "univariate":
        y = df[spec["y_col"]].to_numpy(dtype=np.float32)
        return {"kind": kind, "y": y, "X": None, "y_cols": None, "hist_exog_cols": None}
    if kind == "multivariate":
        cols = list(spec["y_cols"])
        y = df[cols].to_numpy(dtype=np.float32)
        return {"kind": kind, "y": y, "X": None, "y_cols": cols, "hist_exog_cols": None}
    if kind == "covariate":
        y = df[spec["y_col"]].to_numpy(dtype=np.float32)
        cols = list(spec["hist_exog_cols"])
        X = df[cols].to_numpy(dtype=np.float32)
        return {"kind": kind, "y": y, "X": X, "y_cols": None, "hist_exog_cols": cols}
    raise ValueError(f"unknown dataset kind {kind!r}")


def load_dataset_y(spec: dict) -> np.ndarray:
    """Backward-compatible univariate loader (tests / callers)."""
    data = load_dataset(spec)
    if data["kind"] != "univariate":
        raise ValueError("load_dataset_y only supports univariate datasets")
    return data["y"]


def nf_subprocess_code(
    nf_name: str,
    spec: dict,
    h: int,
    input_size: int,
    params: dict,
    seed: int,
    threads: int,
) -> str:
    """Build the `python -c` source run inside .venv-nf for one NF seed.

    torch.set_num_threads pins CPU threads; the neuralforecast import is BEFORE
    t0 so the timer measures only fit+predict (fair timing). Loss is MAE.

    Dataset kinds:
      univariate  — (unique_id, ds, y); n_series=1 when required
      multivariate — N INDEPENDENT univariate fits (one per y_col), metrics
        averaged and one timer around all N — mirroring run_chronax_seed's
        multivariate branch. Chronax is univariate-only by design, so fitting NF
        jointly (n_series=N) would compare N independent models against one
        cross-learning model and measure parameter sharing, not the engine.
      covariate   — single series + hist_exog_list / futr_exog_list when supported
    """
    extra = _nf_extra_dict(params)
    kind = dataset_kind(spec)
    path = spec["path"]
    ds_col = spec["ds_col"]
    freq = spec["freq"]
    name = spec["name"]

    if kind == "multivariate":
        y_cols = list(spec["y_cols"])
        return f'''
import json, time, inspect, numpy as np, pandas as pd
import torch
torch.set_num_threads({int(threads)})
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import {nf_name}
raw = pd.read_csv({path!r})
raw = raw.rename(columns={{ {ds_col!r}: 'ds' }})
raw['ds'] = pd.to_datetime(raw['ds'])
sig = inspect.signature({nf_name}).parameters
maes, smapes = [], []
# PER-SERIES: one independent univariate fit per channel, mirroring the Chronax
# side (run_chronax_seed's multivariate branch). Chronax is univariate-only by
# design, so a joint n_series=N fit here would pit N independent models against
# one cross-learning model — measuring shared parameters, not the engine.
# One timer spans ALL N fits, matching run_chronax_seed's single `elapsed`.
t0 = time.perf_counter()
for col in {y_cols!r}:
    df = raw[['ds', col]].rename(columns={{col: 'y'}}).copy()
    df['unique_id'] = col
    df = df[['unique_id','ds','y']].sort_values('ds').reset_index(drop=True)
    train, test = df.iloc[:-{h}], df.iloc[-{h}:]
    y_true = test['y'].to_numpy()
    kw = {extra}
    if 'n_series' in sig and sig['n_series'].default is inspect.Parameter.empty:
        kw.setdefault('n_series', 1)   # univariate benchmark
    m = {nf_name}(h={h}, input_size={input_size}, loss=MAE(), random_seed={seed},
        accelerator='cpu', enable_progress_bar=False, logger=False,
        enable_model_summary=False, enable_checkpointing=False, **kw)
    nf = NeuralForecast(models=[m], freq={freq!r})
    nf.fit(df=train)
    fcst = nf.predict()
    y_hat = fcst[{nf_name!r}].to_numpy()
    maes.append(float(np.mean(np.abs(y_true - y_hat))))
    denom = (np.abs(y_true) + np.abs(y_hat)) / 2
    smapes.append(float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_hat) / denom)) * 100))
t = time.perf_counter() - t0
print(json.dumps({{'mae': float(np.mean(maes)), 'smape': float(np.mean(smapes)), 'wallclock': t}}))
'''

    if kind == "covariate":
        y_col = spec["y_col"]
        exog_cols = list(spec["hist_exog_cols"])
        return f'''
import json, time, inspect, numpy as np, pandas as pd
import torch
torch.set_num_threads({int(threads)})
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import {nf_name}
df = pd.read_csv({path!r})
df = df.rename(columns={{ {ds_col!r}: 'ds', {y_col!r}: 'y' }})
df['ds'] = pd.to_datetime(df['ds']); df['unique_id'] = {name!r}
keep = ['unique_id','ds','y'] + {exog_cols!r}
df = df[keep].sort_values('ds').reset_index(drop=True)
train, test = df.iloc[:-{h}], df.iloc[-{h}:]
y_true = test['y'].to_numpy()
kw = {extra}
sig = inspect.signature({nf_name}).parameters
if 'hist_exog_list' in sig:
    kw['hist_exog_list'] = {exog_cols!r}
if 'futr_exog_list' in sig:
    kw['futr_exog_list'] = {exog_cols!r}
if 'n_series' in sig and sig['n_series'].default is inspect.Parameter.empty:
    kw.setdefault('n_series', 1)
m = {nf_name}(h={h}, input_size={input_size}, loss=MAE(), random_seed={seed},
    accelerator='cpu', enable_progress_bar=False, logger=False,
    enable_model_summary=False, enable_checkpointing=False, **kw)
nf = NeuralForecast(models=[m], freq={freq!r})
futr_df = None
if 'futr_exog_list' in sig:
    futr_df = test[['unique_id','ds'] + {exog_cols!r}].copy()
t0 = time.perf_counter()
nf.fit(df=train)
fcst = nf.predict(futr_df=futr_df) if futr_df is not None else nf.predict()
t = time.perf_counter() - t0
y_hat = fcst[{nf_name!r}].to_numpy()
mae = float(np.mean(np.abs(y_true - y_hat)))
denom = (np.abs(y_true) + np.abs(y_hat)) / 2
smape = float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_hat) / denom)) * 100)
print(json.dumps({{'mae': mae, 'smape': smape, 'wallclock': t}}))
'''

    # univariate (default)
    y_col = spec["y_col"]
    return f'''
import json, time, inspect, numpy as np, pandas as pd
import torch
torch.set_num_threads({int(threads)})
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import {nf_name}
df = pd.read_csv({path!r})
df = df.rename(columns={{ {ds_col!r}: 'ds', {y_col!r}: 'y' }})
df['ds'] = pd.to_datetime(df['ds']); df['unique_id'] = {name!r}
df = df[['unique_id','ds','y']].sort_values('ds').reset_index(drop=True)
train, test = df.iloc[:-{h}], df.iloc[-{h}:]
y_true = test['y'].to_numpy()
kw = {extra}
sig = inspect.signature({nf_name}).parameters
if 'n_series' in sig and sig['n_series'].default is inspect.Parameter.empty:
    kw.setdefault('n_series', 1)   # univariate benchmark
m = {nf_name}(h={h}, input_size={input_size}, loss=MAE(), random_seed={seed},
    accelerator='cpu', enable_progress_bar=False, logger=False,
    enable_model_summary=False, enable_checkpointing=False, **kw)
nf = NeuralForecast(models=[m], freq={freq!r})
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

sys.path.insert(0, str(REPO))
from benchmarks.neural import metrics, registry  # noqa: E402
from benchmarks.neural.run import load_config, model_params  # noqa: E402


def _dataset_spec(cfg: dict, name: str) -> dict:
    for d in cfg["datasets"]:
        if d["name"] == name:
            spec = dict(d)
            p = Path(spec["path"])
            spec["path"] = str(p if p.is_absolute() else (REPO / p).resolve())
            return spec
    raise KeyError(f"unknown dataset {name!r}")


def _chronax_fit_predict(model, y_train, h, X_train=None, X_test=None):
    """Call Chronax fit/predict, using exog only when the model declares uses_exog.

    TFT: hist exog via ``X=`` plus future-known via ``futr_exog=`` (same columns).
    Informer / MLP / TCN: future-known only via ``futr_exog=`` (``X=`` raises).
    Other uses_exog models: ``fit(y, X=...)`` / ``predict(h, X=...)``.
    """
    import jax.numpy as jnp

    y_train = jnp.asarray(y_train)
    uses_exog = bool(getattr(model, "uses_exog", False))
    if not uses_exog or X_train is None:
        model.fit(y_train)
        return np.asarray(model.predict(h=h)["mean"])

    X_train_j = jnp.asarray(X_train)
    X_test_j = None if X_test is None else jnp.asarray(X_test)
    fit_sig = inspect.signature(model.fit)
    pred_sig = inspect.signature(model.predict)
    has_futr = "futr_exog" in fit_sig.parameters
    alias = getattr(model, "alias", type(model).__name__).lower()
    # Only TFT accepts historical X among the current uses_exog neural ports;
    # Informer / MLP / TCN reject X= and require futr_exog=.
    hist_ok = "tft" in alias

    fit_kwargs = {}
    if has_futr:
        fit_kwargs["futr_exog"] = X_train_j
    if hist_ok and "X" in fit_sig.parameters:
        fit_kwargs["X"] = X_train_j
    if fit_kwargs:
        model.fit(y_train, **fit_kwargs)
    else:
        model.fit(y_train, X=X_train_j)

    pred_kwargs = {}
    if "futr_exog" in pred_sig.parameters and X_test_j is not None:
        pred_kwargs["futr_exog"] = X_test_j
    elif "X" in pred_sig.parameters and X_test_j is not None and hist_ok:
        pred_kwargs["X"] = X_test_j
    return np.asarray(model.predict(h=h, **pred_kwargs)["mean"])


def run_chronax_seed(
    cls: type,
    data: dict,
    h: int,
    input_size: int,
    chronax_params: dict,
    seed: int,
) -> dict:
    """Fit+predict one Chronax seed; return metric row with an `error` field."""
    import jax.numpy as jnp  # noqa: F401 — ensure jax import path for models

    row = {"mae": None, "smape": None, "wallclock_s": None, "error": ""}
    kind = data["kind"]
    try:
        t0 = time.perf_counter()
        if kind == "multivariate":
            y = data["y"]  # (T, N)
            y_train, y_test = y[:-h], y[-h:]
            maes, smapes = [], []
            for i in range(y.shape[1]):
                model = cls(h=h, input_size=input_size, random_seed=seed, **chronax_params)
                pred = _chronax_fit_predict(model, y_train[:, i], h)
                maes.append(metrics.mae(y_test[:, i], pred))
                smapes.append(metrics.smape(y_test[:, i], pred))
            elapsed = time.perf_counter() - t0
            row.update(mae=float(np.mean(maes)), smape=float(np.mean(smapes)),
                       wallclock_s=elapsed)
        elif kind == "covariate":
            y, X = data["y"], data["X"]
            y_train, y_test = y[:-h], y[-h:]
            X_train, X_test = X[:-h], X[-h:]
            model = cls(h=h, input_size=input_size, random_seed=seed, **chronax_params)
            pred = _chronax_fit_predict(model, y_train, h, X_train=X_train, X_test=X_test)
            elapsed = time.perf_counter() - t0
            row.update(mae=metrics.mae(y_test, pred),
                       smape=metrics.smape(y_test, pred), wallclock_s=elapsed)
        else:
            y = data["y"]
            y_train, y_test = y[:-h], y[-h:]
            model = cls(h=h, input_size=input_size, random_seed=seed, **chronax_params)
            pred = _chronax_fit_predict(model, y_train, h)
            elapsed = time.perf_counter() - t0
            row.update(mae=metrics.mae(y_test, pred),
                       smape=metrics.smape(y_test, pred), wallclock_s=elapsed)
    except Exception as e:  # noqa: BLE001
        # Documented broad catch (spec §10): a diverged Chronax fit raises a bare
        # RuntimeError (gru_model.py); a typed exception is a chronax/ change out
        # of scope. Record the message and continue so one seed can't kill a run.
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def run_nixtla_seed(nf_name: str, spec: dict, h: int, input_size: int, nf_params: dict, seed: int, threads: int) -> dict:
    """Spawn a fresh .venv-nf process for one NF seed; return a metric row.

    Fresh process per seed = clean per-seed RNG + matches how baselines were
    captured (spec §7, §14).
    """
    row = {"mae": None, "smape": None, "wallclock_s": None, "error": ""}
    code = nf_subprocess_code(nf_name, spec, h, input_size, nf_params, seed, threads)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(threads))  # pin torch/MKL threads for the NF proc
    try:
        out = subprocess.run([str(NF_VENV_PY), "-c", code],
                             capture_output=True, text=True, check=True, env=env)
        d = json.loads(out.stdout.strip().splitlines()[-1])
        row.update(mae=d["mae"], smape=d["smape"], wallclock_s=d["wallclock"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as e:
        # Typed catch (spec §10): a failed NF subprocess / unparseable JSON becomes
        # an error row; the orchestrator logs and moves on. Not the broad catch.
        detail = getattr(e, "stderr", None) or str(e)
        row["error"] = f"{type(e).__name__}: {detail}"
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
    seeds = args.seeds if args.seeds is not None else exp["seeds"]
    warmup, threads = exp["warmup_seeds"], exp["threads"]
    spec = _dataset_spec(cfg, args.dataset)
    h, input_size = resolve_horizon(spec, exp)
    chronax_params, nf_params = model_params(cfg, args.model)

    # Pin threads BEFORE importing jax/chronax (see module docstring).
    os.environ.update(build_thread_env(threads))

    if args.library == "chronax":
        cls = registry.resolve_chronax(args.model)
        data = load_dataset(spec)
        for i, seed in enumerate(seeds):
            row = run_chronax_seed(cls, data, h, input_size, chronax_params, seed)
            emit_row(args.library, args.dataset, args.model, seed, i, warmup, row)
    else:
        nf_name = registry.nf_model_name(args.model)
        for i, seed in enumerate(seeds):
            row = run_nixtla_seed(nf_name, spec, h, input_size,
                                  nf_params, seed, threads)
            emit_row(args.library, args.dataset, args.model, seed, i, warmup, row)


if __name__ == "__main__":
    main()
