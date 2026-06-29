# Temporal Fusion Transformer (TFT) — Chronax Design Spec

**Date:** 2026-06-29
**Author:** xan
**Status:** Draft for review
**Target branch:** `xan/tft` (off `origin/prod`)

## 1. Goal

Implement the Temporal Fusion Transformer (Lim et al., 2019) in Chronax as a
`flax.nnx` model wrapped in a `BaseForecaster` subclass, **functionally
equivalent to `neuralforecast.TFT`** but faster via XLA-compiled
training/inference. "Functional equivalence" here is **architectural +
behavioral parity** (same blocks, defaults, torch init, robust scaler, window
sampling, and the multi-quantile loss mechanism), verified by a Chronax-vs-NF
benchmark — **not** bit-exact forward parity.

## 2. Decisions (resolved during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Exogenous support | **Full: static + historical + future-known** | TFT's signature capability; required for true NF equivalence. First exog-capable neural model in Chronax. |
| Framework / integration | **`flax.nnx` + `BaseForecaster` subclass** (iTransformer pattern) | Newest repo pattern; inherits conformal intervals, pickle, and `vmap`-safe `nnx.scan` training. `nnx.LSTMCell`/`GRUCell` confirmed available in flax 0.10.7. |
| Parity bar | **Architectural + behavioral** | Matches iTransformer/Autoformer standard; faster wall-clock; bit-exact is out of scope. |
| Probabilistic output | **Native multi-quantile, first-class** (default still `mae`) | The paper's headline feature; NF supports it via `loss=MQLoss(...)`. Default `loss='mae'` keeps out-of-the-box behavior == NF. |

## 3. Architecture

A faithful port of `neuralforecast/models/tft.py`. Every block is a thin `nnx`
module. Two primitives (GLU, GRN) compose into everything else.

### 3.1 Primitives (`tft_layers.py`)

- **GLU** `Linear(h, 2h)` → `jax.nn.glu` (== `σ(a) ⊙ b`).
- **GRN(a, c=None)**:
  ```
  x = lin_a(a)
  if c is not None: x = x + lin_c(c)          # context broadcast over time; lin_c has no bias
  x = activation(x)                            # default ELU
  x = lin_i(x)
  x = dropout(x)
  x = GLU(x)
  residual = a if out_proj is None else out_proj(a)
  return LayerNorm(x + residual)              # MaybeLayerNorm: Identity if output_size == 1
  ```
- **TFTEmbedding**: each continuous feature `f` → `f[..., None] * emb_vec + emb_bias`
  (per-feature learned vector + bias). NF treats all exog as continuous; **no
  categorical embedding tables** (matches NF's TFT path).
- **VariableSelectionNetwork(x, context=None)**:
  ```
  Xi = flatten(x, last 2 dims)                          # [..., num_inputs*hidden]
  weights = softmax(joint_grn(Xi, c=context))           # [..., num_inputs]
  transformed = stack([var_grn_i(x[..., i, :])], -1)    # [..., hidden, num_inputs]
  return (transformed @ weights[..., None]).squeeze(-1), weights
  ```
- **InterpretableMultiHeadAttention**:
  - `qkv = Linear(h, (2*n_head + 1)*d_head, bias=False)`; split into Q,K (multi-head)
    and a **single shared V** (not per-head).
  - `scores = (Q·Kᵀ)·d_head^-0.5`; causal mask via `jnp.tril` (computed on the fly,
    **not** a stored buffer — keeps it `vmap`/`scan`-pure); softmax; attn-dropout.
  - `ctx = attn @ V` (V broadcast over heads); **mean over heads**; `out_proj`.

### 3.2 Encoders / decoder (`tft_module.py`)

- **StaticCovariateEncoder(s)**: VSN over static vars → `variable_ctx`; then context
  GRNs produce `cs` (variable-selection ctx), `ce` (enrichment ctx), and `ch`/`cc`
  (per-layer LSTM hidden/cell init states). For GRU, `cc = ch`. If no static exog,
  all four are zeros and `static_encoder` is skipped.
- **TemporalCovariateEncoder(hist, futr, cs, ch, cc)**:
  ```
  hist_feat, hist_w = history_vsn(hist, context=cs)
  enc_out, state = LSTM_encoder(hist_feat, init=(cc, ch))     # nnx.scan over nnx.LSTMCell
  futr_feat, futr_w = future_vsn(futr, context=cs)
  dec_out, _ = LSTM_decoder(futr_feat, init=state)
  feat = concat([hist_feat, futr_feat], time)                 # input embedding (residual)
  temporal = concat([enc_out, dec_out], time)
  temporal = LayerNorm(GLU(temporal) + feat)                  # input gate + skip
  return temporal, hist_w, futr_w
  ```
  Separate encoder/decoder cells (NF `history_encoder`/`future_encoder`).
  `rnn_type ∈ {lstm, gru}`, `n_rnn_layers` default 1 (stack supported via per-layer
  `ch`/`cc`).
- **TemporalFusionDecoder(temporal, ce)**:
  ```
  enriched = enrichment_grn(temporal, c=ce)
  x, attn_w = attention(enriched, causal=True)
  x = x[:, input_size:, :];  enriched = enriched[:, input_size:, :];  temporal = temporal[:, input_size:, :]
  x = LayerNorm(GLU(x) + enriched)
  x = positionwise_grn(x)
  x = LayerNorm(GLU(x) + temporal)
  return x, attn_w
  ```

### 3.3 Top-level `TFTNet.__call__(windows_batch) -> [B, h, multiplier]`

Mirrors NF `TFT.forward`. Inputs come from a `windows_batch` dict:
`insample_y [B,L,1]`, `hist_exog [B,L,H]`, `futr_exog [B,L+h,F]`, `stat_exog [B,S]`.

```
embed all inputs (TFTEmbedding)
if stat: cs,ce,ch,cc = static_encoder(s_emb) else zeros
historical = concat([futr_emb[:,:L], hist_emb[:,:L], tgt_emb[:,:L]], var-axis)
future     = futr_emb[:, L:]
temporal, hw, fw = temporal_encoder(historical, future, cs, ch, cc)
temporal, aw     = temporal_fusion_decoder(temporal, ce)
y_hat = output_adapter(temporal)                 # Linear(hidden, outputsize_multiplier)
return y_hat                                      # [B, h, multiplier]
```
`num_historic_vars = max(futr_size, 1) + hist_size + tgt_size`. If `futr_exog` is
None, NF substitutes the last insample value repeated over `L+h`; we replicate.
Interpretability weights (`hw`, `fw`, `aw`, static weights) are computed but not
surfaced in v1's forecast dict (available for a later `interpret()` method).

## 4. Exogenous data model & windowing (`tft_training.py`)

Three covariate kinds with different time extents:
- **static** `[S]` — no time axis (per series).
- **historical** `[T, H]` — observed up to `t`, aligned with `y` (NF `hist_exog`).
  Feeds **only** the encoder (never the decoder), so no future values are required.
- **future-known** — known across input **and** horizon (NF `futr_exog`). Conceptually
  defined over `[T+h]`; in practice supplied as the **`[T, F]`** portion aligned with
  `y` at fit (each training window's horizon falls *within* the observed `T`, so no
  out-of-sample values are needed during training) and the **`[h, F]`** horizon portion
  at predict (the model concatenates the trailing-`L` history with it to form the
  `[L+h]` window).

`build_windows` extends the iTransformer windower to carry exog, producing per
window: `insample_y [L]`, `hist_exog [L, H]`, `futr_exog [L+h, F]`, `stat_exog [S]`,
plus the `[h]` target. Window sampling replicates NF's regime-dependent scheme
(`with replacement` when `n_windows < windows_batch_size`, else a
without-replacement permutation) — load-bearing for accuracy parity (see
iTransformer `train`).

## 5. Forecaster API (`tft_model.py`)

`class TFT(BaseForecaster)`, `uses_exog = True`. Extends the base signature with
optional exog kwargs (the base explicitly permits this):

```python
fit(y, X=None, *, futr_exog=None, stat_exog=None) -> self
predict(h, X=None, *, futr_exog=None, level=None) -> dict
forecast(y, h, X=None, X_future=None, *, futr_exog=None, stat_exog=None,
         level=None, fitted=False) -> dict
```
- `X` == historical-only exog over `y` `(T,H)`; `futr_exog` == future-known exog
  history `(T,F)` at fit and the horizon `(h,F)` at predict; `stat_exog` `(S,)`.
  `predict`'s `X` is accepted for base-signature compatibility but unused (hist exog
  for the input window is cached from fit; it has no future values).
- **`conformity_scores` threading (decides the conformal path).** The base calls
  `forecast(h=h, y=y_train, X=X_train, X_future=X_test)`, splitting a *single* padded
  exog matrix into a history part (`X_train`) and a horizon part (`X_test`). Only an
  exog kind that has both — i.e. **future-known** — can flow through it; `stat_exog`
  is carried as a fit-time attribute. So in the conformal path TFT reads `X` as the
  future-known history and `X_future` as its horizon. Historical-only exog cannot be
  threaded by `conformity_scores` (it has no future rows) ⇒ the conformal
  `predict(level=…)` path supports **future-known + static** exog only (see §7.3). The
  exact attribute names / fit-context caching are finalized in the implementation plan.
- Per-window **RobustScaler** (median/MAD, 0.6745·std fallback) on temporal channels
  (NF `scaler_type='robust'` default); static raw. Reuses the `kan_scaler` pattern in
  a self-contained `tft_scaler.py`.
- Pickle via `__getstate__`/`__setstate__` (`nnx.split`/`nnx.update`), mirroring
  iTransformer.

### 5.1 Interval routing
- **point loss** (`mae`/`mse`/`huber`): `predict` returns `{"mean"}`; `level=…`
  adds conformal `lo-XX`/`hi-XX` via the inherited `BaseForecaster` path.
- **quantile loss** (`MultiQuantileLoss`): `"mean"` = the q=0.5 head; `level=[80]`
  → quantile pair `(0.1, 0.9)` → `lo-80`/`hi-80` **emitted directly by the model**.
  Requested levels must map to trained quantiles (else a clear error). Optional
  monotonic sort prevents inverted intervals (NF does not enforce monotonicity).

## 6. Loss & defaults (`tft_losses.py`)

- Registry: `mae` (default), `mse`, `huber`, each with `outputsize_multiplier = 1`.
- `MultiQuantileLoss(quantiles=[0.1, 0.5, 0.9])` — picklable callable computing
  `mean_q Σ QL(y, ŷ_q, q)`, `QL = q·(y−ŷ)₊ + (1−q)·(ŷ−y)₊`,
  `outputsize_multiplier = len(quantiles)`. Quantiles are sorted and **must include
  0.5** (the `"mean"`/median head); enforced at construction. Default `[0.1, 0.5, 0.9]`
  gives P10/P50/P90 — the paper's headline output and `level=[80]`.
- `loss=` accepts a registry string **or** a `MultiQuantileLoss` instance.
- **NF-exact defaults:** `hidden_size=128, n_head=4, attn_dropout=0.0, dropout=0.1,
  grn_activation='ELU', n_rnn_layers=1, rnn_type='lstm', learning_rate=1e-3,
  max_steps=1000, windows_batch_size=1024, scaler_type='robust', loss='mae'`.
  `input_size`, `h` required. Torch-style init via the repo's `_TorchLinearInit`;
  ELU/ReLU/etc. activations per `grn_activation`.

## 7. Speed strategy & tradeoffs (to report)

**Speedup sources:** whole fwd+bwd JIT-fused by XLA; the entire training loop is one
`nnx.scan` (no Python per-step dispatch, no PyTorch-Lightning/dataloader overhead);
windows and CV via `vmap`.

**Tradeoffs (explicit):**
1. **The LSTM is the soft spot in the speed claim.** `lax/nnx.scan` is XLA-fused but
   inherently O(T) sequential; PyTorch's cuDNN LSTM is a hand-tuned kernel, so on
   large-batch GPU the recurrence may not beat cuDNN. Net win is on compilation
   amortization, CPU/CV workloads, and the (parallel) non-LSTM blocks.
2. **Not bit-exact.** Different reduction order + LSTM kernel ⇒ small numeric
   divergence from NF is expected and accepted. Verified by benchmark +
   beats-naive / comparable-accuracy / exog-improves-accuracy tests, not a `max|Δ|`
   gate.
3. **Conformal + historical-only exog caveat.** `conformity_scores` slices
   `X_future` from the horizon region of `X` — i.e. it assumes exog is known in the
   future. The conformal `predict(level=…)` path is clean for **future-known +
   static** exog; with historical-only exog it would treat it as future-known.
   Documented limitation (use a quantile loss for native intervals instead).
4. First-call JIT compile latency on one-off forecasts (standard JAX).

## 8. File layout

```
chronax/models/tft/
  __init__.py          # exports TFT
  tft_layers.py        # GLU, GRN, VariableSelectionNetwork, InterpretableMultiHeadAttention, TFTEmbedding, _TorchLinearInit
  tft_module.py        # StaticCovariateEncoder, TemporalCovariateEncoder, TemporalFusionDecoder, TFTNet
  tft_losses.py        # mae/mse/huber + MultiQuantileLoss; resolve()
  tft_scaler.py        # RobustScaler / IdentityScaler (self-contained)
  tft_training.py      # exog-aware build_windows, forward_loss, nnx.scan train(), predict_step()
  tft_model.py         # TFT(BaseForecaster)
tests/test_tft.py
benchmarks/tft_benchmark.py   # Chronax vs NF, NF defaults (accuracy + wall-clock)
```
Register `from .tft import TFT` + add `"TFT"` to `__all__` in
`chronax/models/__init__.py`.

## 9. Testing plan (`tests/test_tft.py`, mirrors `test_itransformer.py`)

- **Layers:** GLU gating; GRN residual + context injection + MaybeLayerNorm identity;
  VSN weights sum to 1 and shape; interpretable-MHA shared-V + head-averaging +
  causal mask (future cannot attend to past-of-future); torch-init bounds.
- **Module:** LSTM seq2seq state threading (decoder continues encoder state); static
  context init; full `TFTNet` shapes for all exog combinations (all three / none /
  each subset); `outputsize_multiplier` for point vs quantile.
- **Training:** exog-aware windowing shapes; sampling regimes; loss decreases;
  determinism with fixed seed; divergence raises.
- **Model (BaseForecaster conformance):** `fit`/`predict`/`forecast`; `uses_exog`;
  point→conformal and quantile→native interval keys; `level` subset validation;
  monotonic-sort option; pickle round-trip (params + loss); `vmap`/conformity_scores
  finite; beats-naive on an easy signal; **exog actually improves accuracy** on a
  signal driven by a known future covariate; edge cases (constant series, h=1,
  short-series raise, 2-D input raise).
- **Namespace:** importable from `chronax.models`.

## 10. Benchmark

`benchmarks/` script comparing Chronax TFT vs `neuralforecast.TFT` at NF defaults on
standard series, reporting accuracy deltas and wall-clock (fit + predict), in the
style of the existing KAN/iTransformer benchmarks. Resumable, results to
`benchmark_results/`.

## 11. Out of scope (v1)

- Categorical-embedding exog (NF's TFT path uses continuous only — matched).
- `interpret()` surfacing attention/VSN weights (computed internally; expose later).
- Distribution losses beyond quantile (`DistributionLoss`/PMM/GMM).
- Multi-layer LSTM tuning beyond `n_rnn_layers` plumbing.
- Bit-exact weight-port parity gate.

## 12. Open questions / risks

- **R1 — nnx LSTM carry convention.** `nnx.LSTMCell` carry order is `(c, h)`; must
  match NF's `(h, c)` init mapping from `ch`/`cc`. Verify in an isolated test before
  wiring the encoder.
- **R2 — quantile crossing.** Mitigated by optional monotonic sort at predict.
- **R3 — exog scaling fidelity.** NF applies the temporal scaler to continuous exog
  too (`TemporalNorm`). v1 applies robust per-window scaling to temporal channels
  (incl. exog) and leaves static raw; confirm this matches NF closely in the
  benchmark.
- **R4 — module file size.** If `tft_module.py` grows beyond ~500 lines, the
  layers/module split already isolates primitives; revisit further splitting then.
