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


def nf_subprocess_code(nf_name, spec, h, input_size, params, seed, threads) -> str:
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
