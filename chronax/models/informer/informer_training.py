"""Window construction and JIT/scan training for Informer.

Windows are scaled per-window (scaler stats on the insample target; per-channel
scaling on future-known exog), the loss is computed in scaled space, and the
whole training loop is one ``nnx.scan`` so it stays ``vmap``-traceable for
``BaseForecaster.conformity_scores`` (mirrors the TFT/iTransformer trainers).
Each scan step draws a fresh ProbSparse ``sample_key`` -- split off a stream
independent from the batch-index keys -- that ``InformerNet`` fans out to every
attention site (``encoder_layers + 2*decoder_layers`` keys per call), so
reshuffling the training batches never perturbs which ProbSparse queries get
refined. Inference (``_forward_det``/``predict_step``) instead forwards with a
fixed ``jax.random.PRNGKey(0)`` created INSIDE the jitted function -- never
hoisted out or stored -- so repeated predictions, pickled round-trips, and
conformal windows are bit-identical.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import chronax.models.informer.informer_losses as _losses
from chronax.models.informer.informer_losses import LossFn, MultiQuantileLoss
from chronax.models.informer.informer_module import InformerNet


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NF-parity rolling windows over ``y`` right-padded with ``h`` zeros.

    NF pads the training series with ``ConstantPad1d((0, h), 0)`` before
    windowing, keeping every window with >=1 valid target point and masking the
    padded tail out of the loss. Those partial windows put the insample contexts
    ending at the very last observations into training; a full-window-only form
    drops exactly those newest contexts, which costs accuracy on trending series.

    Returns ``(windows [n, input_size+h], target_mask [n, h])`` with
    ``n = len(y) - input_size``; mask is 1.0 where the target position is a
    real observation and 0.0 in the zero-padded tail.
    """
    T = y.shape[0]
    n = T - input_size
    if n <= 0:
        raise ValueError(f"Series length {T} too short for input_size={input_size}.")
    window = input_size + h
    y_pad = jnp.concatenate([y, jnp.zeros((h,), y.dtype)])
    idx = jnp.arange(window)[None, :] + jnp.arange(n)[:, None]
    target_mask = (idx[:, input_size:] < T).astype(y.dtype)
    return y_pad[idx], target_mask


def build_exog_windows(arr: jnp.ndarray, input_size: int, h: int, n_windows: int, span: str) -> jnp.ndarray:
    """Rolling windows of an exog array ``[T, F]`` (right-padded with ``h`` zero
    rows, mirroring `build_windows` — NF pads the whole temporal tensor, so late
    windows see zeros in the padded region of exog channels too).

    ``span="input"`` -> ``[n, input_size, F]`` (encoder window); ``span="full"`` ->
    ``[n, input_size+h, F]`` (future-known spanning input + horizon).
    """
    length = input_size if span == "input" else input_size + h
    arr_pad = jnp.concatenate([arr, jnp.zeros((h, arr.shape[1]), arr.dtype)])
    idx = jnp.arange(length)[None, :] + jnp.arange(n_windows)[:, None]
    return arr_pad[idx]


def _scale_exog(windows: jnp.ndarray, scaler, stats_len: int | None = None) -> jnp.ndarray:
    """Per-channel per-window robust scaling of ``[B, T, F]`` exog.

    ``stats_len`` restricts the STATISTICS to the first ``stats_len`` positions
    (the insample span) while transforming the whole window — NF's
    ``_normalization`` masks the horizon out of the scaler stats. ``None`` keeps
    full-span stats (insample-span windows only).
    """
    stats_src = windows if stats_len is None else windows[:, :stats_len]
    shift, scale = scaler.stats(stats_src, axis=1)      # [B, 1, F]
    return scaler.transform(windows, shift, scale)


# Elementwise forms of the registry point losses, for NF-parity masked reduction
# (NF losses compute sum(loss*mask)/sum(mask) — `_weighted_mean`). Keyed by the
# registry function OBJECTS so a user's custom callable that happens to share a
# name falls through to the custom branch instead of being shadowed.
_ELEMENTWISE = {
    _losses.mae: lambda e: jnp.abs(e),
    _losses.mse: lambda e: e * e,
    _losses.huber: lambda e: jnp.where(jnp.abs(e) <= 1.0, 0.5 * e * e, jnp.abs(e) - 0.5),
}


def forward_loss(net, y_windows, target_mask=None, *, h, input_size, scaler, loss_fn,
                 futr_windows=None, sample_key, deterministic=False):
    """Scale, forward, and reduce a point/quantile loss in scaled space.

    ``target_mask [B, h]`` marks real target positions (0 in the NF h-padded
    tail); the loss is the masked mean over valid elements, matching NF's
    ``_weighted_mean``. ``None`` means all-valid.
    """
    insample = y_windows[:, :input_size]                # [B, L]
    target = y_windows[:, input_size:]                  # [B, h]
    shift, scale = scaler.stats(insample, axis=1)       # [B, 1]
    insample_z = scaler.transform(insample, shift, scale)[..., None]   # [B, L, 1]
    target_z = scaler.transform(target, shift, scale)   # [B, h]
    futr_z = (_scale_exog(futr_windows, scaler, stats_len=input_size)
              if futr_windows is not None else None)
    pred = net(insample_z, futr_exog=futr_z, sample_key=sample_key,
               deterministic=deterministic, use_running_average=deterministic)   # [B, h, mult]
    if target_mask is None:
        target_mask = jnp.ones_like(target_z)
    denom = jnp.sum(target_mask)
    if isinstance(loss_fn, MultiQuantileLoss):
        q = jnp.asarray(loss_fn.quantiles, dtype=pred.dtype)          # [Q]
        err = target_z[..., None] - pred                               # [B, h, Q]
        ql = jnp.maximum(q * err, (q - 1.0) * err)
        # NF quirk kept: MQLoss's 1/len(quantiles) hits a [1,1,1,Q] tensor
        # (len==1), so NF SUMS over quantiles and means over valid positions.
        return jnp.sum(ql * target_mask[..., None]) / denom
    ew = _ELEMENTWISE.get(loss_fn)
    if ew is not None:
        return jnp.sum(ew(pred[..., 0] - target_z) * target_mask) / denom
    # Custom callable: zero out masked errors; denominator stays the callable's own.
    return loss_fn(jnp.where(target_mask > 0, pred[..., 0], target_z), target_z)


def _finite_or_raise(losses: jnp.ndarray) -> jnp.ndarray:
    """Divergence guard. No-ops under a higher trace (e.g. conformity_scores's vmap),
    where ``losses`` is a tracer and cannot be concretized to raise a Python error."""
    if isinstance(losses, jax.core.Tracer):
        return losses
    if not bool(jnp.all(jnp.isfinite(losses))):
        i = int(jnp.argmax(~jnp.isfinite(losses)))
        raise RuntimeError(
            f"Non-finite loss at step {i}. Training diverged. Lower learning_rate "
            "or windows_batch_size, or check the series for extreme values."
        )
    return losses


def train(net, y, *, h, input_size, max_steps, windows_batch_size, lr, seed, loss_fn, scaler,
          futr_exog=None):
    """Train ``net`` in place via one ``nnx.scan``. Returns per-step losses.

    Two independent RNG streams are split off ``seed``: batch-index keys (NF's
    regime-dependent window sampling, below) and ProbSparse ``attn_keys`` (one
    per scan step, threaded through to every attention site inside
    ``InformerNet``) -- so shuffling the training batches never perturbs which
    ProbSparse queries get sampled for a given step.
    """
    y_windows, target_mask = build_windows(y, input_size, h)
    n = y_windows.shape[0]
    futr_w = build_exog_windows(futr_exog, input_size, h, n, "full") if futr_exog is not None else None
    batch_key, attn_key = jax.random.split(jax.random.PRNGKey(seed))
    step_keys = jax.random.split(batch_key, max_steps)
    if n < windows_batch_size:                          # NF: torch.randint -> with replacement
        sample = lambda k: jax.random.choice(k, n, shape=(windows_batch_size,), replace=True)
    else:                                               # NF: torch.randperm[:B] -> without replacement
        sample = lambda k: jax.random.permutation(k, n)[:windows_batch_size]
    batch_idx = jax.vmap(sample)(step_keys)               # [max_steps, B]
    attn_keys = jax.random.split(attn_key, max_steps)     # [max_steps, 2]
    optimizer = nnx.Optimizer(net, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, xs):
        net, opt = carry
        idx, akey = xs
        yb = y_windows[idx]
        mb = target_mask[idx]
        fb = futr_w[idx] if futr_w is not None else None
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, yb, mb, h=h, input_size=input_size, scaler=scaler,
                                   loss_fn=loss_fn, futr_windows=fb,
                                   sample_key=akey, deterministic=False))(net)
        opt.update(grads)
        return (net, opt), loss

    _, losses = step((net, optimizer), (batch_idx, attn_keys))
    return _finite_or_raise(losses)


@nnx.jit
def _forward_det(net, insample_z, futr_z):
    """Inference forward: fixed sampling key (locked decision), frozen BatchNorm stats."""
    return net(insample_z, futr_exog=futr_z, sample_key=jax.random.PRNGKey(0),
               deterministic=True, use_running_average=True)


def predict_step(net, y_context, *, h, input_size, scaler, futr_full=None):
    """Forecast next ``h`` steps from the final ``input_size`` of the series.

    Returns ``[h, multiplier]`` in the **original** scale. ``futr_full`` is the
    ``[input_size+h, F]`` future-known window (history + horizon).
    """
    insample = y_context[None, :]                       # [1, L]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)[..., None]   # [1, L, 1]
    futr_z = (_scale_exog(futr_full[None], scaler, stats_len=input_size)
              if futr_full is not None else None)
    pred_z = _forward_det(net, insample_z, futr_z)[0]  # [h, mult]
    return scaler.inverse(pred_z, shift[0, 0], scale[0, 0])
