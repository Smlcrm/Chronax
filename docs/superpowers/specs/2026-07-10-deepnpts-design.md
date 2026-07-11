# DeepNPTS — JAX/Flax-NNX port into Chronax

**Status:** Design approved, pending implementation
**Date:** 2026-07-10
**Reference:** `neuralforecast.models.deepnpts.DeepNPTS`
**Paper:** Rangapuram, Gasthaus, Stella, Flunkert, Salinas, Wang, Januschowski (2023),
"Deep Non-Parametric Time Series Forecaster", arXiv:2312.14657
**PR base:** `dev` (Chronax convention)

## Summary

DeepNPTS (Deep Non-Parametric Time Series forecaster) is a baseline deep model.
It reads the lookback window through a small MLP that emits, per horizon step, a
softmax weight over each position of the window; the forecast is the
weighted sum of the in-sample values. It is a *learned non-parametric resample*
of the context, not a parametric-distribution head. It is a **point** forecaster
and **univariate**.

This is a faithful univariate, no-exogenous JAX/Flax-NNX port, following the
established Chronax convention used by `itransformer`, `bitcn`, and
`vanilla_transformer`: flat 4-file layout, `BaseForecaster` integration, Optax
`adam`, pluggable point loss, opt-in Box-Cox, conformal prediction intervals,
pickle round-trip, and a NeuralForecast parity benchmark.

## Model

### Reference forward (NF, general)

NF's `forward` handles exogenous inputs and produces, for input `insample_y`
`[B, L, 1]`:

1. Concatenate historic/future/static exogenous, flatten to `[B, input_dim]`.
2. `deepnptsnetwork` (an MLP) maps `[B, input_dim] -> [B, L*h]`.
3. Reshape to `[B, L, h]`, softmax over the **L (window)** axis.
4. Multiply by `insample_y` `[B, L, 1]`, sum over `L` -> `[B, h, 1]`.

The MLP is `n_layers` blocks of `Linear -> ReLU -> [BatchNorm1d] -> [Dropout]`,
then a final `Linear(hidden_size, input_size*h)`.

### Univariate, no-exog collapse

With no exogenous inputs (the Chronax target case, matching the other ports),
`input_dim = input_size` and the forward reduces to:

```
x        = insample_y                 # [B, L, 1] -> reshape [B, L]
z        = MLP(x)                      # n_layers blocks -> [B, hidden]
w        = Linear(hidden, L*h)(z)      # -> reshape [B, L, h]
w        = softmax(w, axis=L-axis)     # weights over the window
forecast = sum(w * insample_y, axis=L) # [B, h] -> [B, h, 1]
```

The softmax-over-`L` structure gives a testable invariant: for every batch item
and horizon step the weights sum to 1, so each forecast value lies within
`[min(window), max(window)]`.

## Files (flat 4-file convention)

`chronax/models/deepnpts/`

- **`deepnpts_module.py`** — `DeepNPTSNet(nnx.Module)`.
  - `_TorchLinearInit` (torch `nn.Linear` default `U(-1/sqrt(fan_in), +...)`),
    reused verbatim from the bitcn convention.
  - MLP blocks: `nnx.Linear -> relu -> [nnx.BatchNorm] -> [nnx.Dropout]`,
    gated on `batch_norm` / `dropout > 0`.
  - Final `nnx.Linear(hidden_size, input_size * h)`.
  - Forward: reshape, MLP, reshape weights `[B, L, h]`, `jax.nn.softmax` over
    the `L` axis, weighted sum with `insample_y` -> `[B, h, 1]`.
  - `deterministic` flag threads dropout; BatchNorm reads `use_running_average
    = deterministic`.
- **`deepnpts_losses.py`** — self-contained `mae` / `mse` / `huber` registry
  plus `resolve`, identical in shape to `bitcn_losses.py`. NF default = MAE.
- **`deepnpts_training.py`** — `build_windows`, `forward_loss`, `train`
  (single `nnx.scan`, per-step window sampling replicating NF's
  replace-vs-permute regime), `predict_step` (jitted deterministic forward).
  The carried `nnx.Optimizer`/model in the scan carries BatchStat too, so
  running stats update in place during training.
- **`deepnpts_model.py`** — `DeepNPTS(BaseForecaster)`:
  `fit`, `predict`, `forecast`, `_compute_fitted_values`, `__getstate__` /
  `__setstate__` (full `nnx.split`, which captures params **and** BatchStat),
  opt-in Box-Cox (`_boxcox` / `_inv_boxcox` / `_select_boxcox_lambda` reused
  from the bitcn convention), conformal intervals via `conformal_params`.

## Hyperparameters

Mirror NF `DeepNPTS` defaults for the univariate case, with one documented
override:

| Param | Default | Notes |
|-------|---------|-------|
| `h` | (required) | forecast horizon |
| `input_size` | `-1 -> 3*h` | lookback |
| `hidden_size` | `32` | NF default |
| `n_layers` | `2` | NF default |
| `dropout` | `0.1` | NF default |
| `batch_norm` | **`False`** | **override** of NF `True`; documented like `use_boxcox` |
| `use_boxcox` | `False` | opt-in variance stabilization (strictly positive) |
| `max_steps` | `1000` | NF default |
| `learning_rate` | `1e-3` | scalar or `optax.ScalarOrSchedule` |
| `windows_batch_size` | `1024` | NF default |
| `random_seed` | `1` | NF default |
| `loss` | `"mae"` | registry name or callable |
| `alias` | `"DeepNPTS"` | |

`batch_norm` defaults to `False` because BatchNorm adds non-parameter running
state (`nnx.BatchStat`) that complicates the training scan, the train/predict
`use_running_average` toggle, and pickle. When enabled it is faithful: momentum
`0.9` (flax decay convention, equivalent to torch's `0.1`) and `eps=1e-5`
(torch default).

## Registry & docs

- `chronax/models/__init__.py`: add
  `from .deepnpts.deepnpts_model import DeepNPTS` and register `"DeepNPTS"` in
  `__all__`, alongside the other deep-learning forecasters.
- Docs: add a DeepNPTS entry under Deep Learning (mirroring the BiTCN docs
  addition) in `docs/models.rst`.

## Tests & benchmark (full scope)

- **`tests/test_deepnpts.py`** — unit + parity:
  - shape contracts (`[B, h, 1]` forward, `(h,)` predict);
  - softmax-weights invariant: forecast in `[min(window), max(window)]`;
  - `fit` -> `predict` happy path and `h`-bounds (`1 <= h <= self.h`);
  - pickle round-trip identity of predictions;
  - Box-Cox strictly-positive guard;
  - `batch_norm=True` path: shape + finiteness only (not strict numerics);
  - **strict parity** vs `neuralforecast.DeepNPTS` at **`batch_norm=False`**,
    `scaler_type="identity"`, `dropout=0.0` in eval, matched init and seed.
- **Benchmark**: DeepNPTS is picked up automatically by the shared neural
  harness (`benchmarks/neural/run.py --models DeepNPTS`), which auto-discovers
  any `chronax.models` forecaster taking `h`/`input_size`/`random_seed`. A
  per-model override in `benchmarks/neural/config.yaml` pins the NF side to
  `scaler_type="identity"` + `batch_norm=False` so the comparison isolates the
  engine, not the preprocessing. (No standalone benchmark script — the
  auto-discovery harness superseded the per-model scripts.)

## Parity notes

- torch `nn.Linear` default init reproduced by `_TorchLinearInit`.
- ReLU and softmax are exact; softmax is taken over the **window (L)** axis.
- `float32` throughout.
- Strict numerical parity is pinned at `batch_norm=False`. The BatchNorm-on path
  is validated for shape/finiteness only, because torch<->flax running-stat
  momentum/eps translation is failure-prone under exact comparison.

## Out of scope (YAGNI)

- Exogenous variables (historic/future/static) — every Chronax deep port is
  univariate no-exog; `fit(X=...)` raises `NotImplementedError`.
- Multivariate output.
- Probabilistic / quantile losses — NF `DeepNPTS` itself rejects non-point
  losses; the port mirrors that.
