"""Window construction and JIT/scan training for MLP.

Windows are scaled per-window (scaler stats on the insample target; per-channel
scaling on future-known exog) and the whole training loop is one ``nnx.scan``
so it stays ``vmap``-traceable for ``BaseForecaster.conformity_scores``. Point
and multi-quantile losses are computed in scaled space; distribution losses
(``GMM``) are evaluated against the ORIGINAL-scale target with the predicted
parameters mapped out of scaled space via ``scale_decouple`` — the network
optimizes in the scaler's normalized range while the likelihood lives in data
units. The MLP forward is fully deterministic (no dropout), so the only RNG
stream is the batch-index sampling; inference needs no key, and repeated
predictions, pickled round-trips, and conformal windows are bit-identical by
construction.

``train`` = ``build_windows`` + ``train_on_windows``; the second half accepts
prebuilt (possibly pooled) window arrays so a hierarchical caller can
cross-learn one net over many series' windows.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import chronax.models.mlp.mlp_losses as _losses
from chronax.models.mlp.mlp_losses import LossFn, MultiQuantileLoss
from chronax.models.mlp.mlp_module import MLPNet


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rolling windows over ``y``, right-padded with ``h`` zeros.

    Padding keeps every window with at least one real target and lets the newest
    observations appear as training contexts; the padded tail is masked out of
    the loss (matters on trending series, where the most recent regime is the
    forecast-relevant one). Returns ``(windows [n, input_size+h], target_mask
    [n, h])`` with ``n = len(y) - input_size``; the mask is 1.0 at real target
    positions and 0.0 in the zero-padded tail.
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
    """Rolling windows of an exog array ``[T, F]``, right-padded with ``h`` zero
    rows to match ``build_windows`` (late windows see zeros in the padded tail of
    each exog channel).

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
    (the insample span) while transforming the whole window, so exog stats never
    see the horizon slice. ``None`` keeps full-span stats (used only for
    insample-span windows).
    """
    stats_src = windows if stats_len is None else windows[:, :stats_len]
    shift, scale = scaler.stats(stats_src, axis=1)      # [B, 1, F]
    return scaler.transform(windows, shift, scale)


# Elementwise forms of the registry point losses, for masked reduction
# (sum(loss*mask)/sum(mask)). Keyed by the registry function OBJECTS so a user's
# custom callable that happens to share a name falls through to the custom
# branch instead of being shadowed.
_ELEMENTWISE = {
    _losses.mae: lambda e: jnp.abs(e),
    _losses.mse: lambda e: e * e,
    _losses.huber: lambda e: jnp.where(jnp.abs(e) <= 1.0, 0.5 * e * e, jnp.abs(e) - 0.5),
}


def forward_loss(net, y_windows, target_mask=None, *, h, input_size, scaler, loss_fn,
                 futr_windows=None):
    """Scale, forward, and reduce the training loss.

    ``target_mask [B, h]`` marks real target positions (0 in the h-padded tail);
    ``None`` means all-valid. Point/quantile losses are masked means in scaled
    space. Distribution losses are the masked NLL of the ORIGINAL-scale target
    under parameters decoupled with the window's own shift/scale.
    """
    insample = y_windows[:, :input_size]                # [B, L]
    target = y_windows[:, input_size:]                  # [B, h]
    shift, scale = scaler.stats(insample, axis=1)       # [B, 1]
    insample_z = scaler.transform(insample, shift, scale)[..., None]   # [B, L, 1]
    futr_z = (_scale_exog(futr_windows, scaler, stats_len=input_size)
              if futr_windows is not None else None)
    pred = net(insample_z, futr_exog=futr_z)            # [B, h, mult]
    if target_mask is None:
        target_mask = jnp.ones_like(target)
    if getattr(loss_fn, "is_distribution_output", False):
        distr_args = loss_fn.domain_map(pred)
        distr_args = loss_fn.scale_decouple(distr_args, loc=shift[..., None],
                                            scale=scale[..., None])
        return loss_fn(target, distr_args, mask=target_mask)
    target_z = scaler.transform(target, shift, scale)   # [B, h]
    denom = jnp.sum(target_mask)
    if isinstance(loss_fn, MultiQuantileLoss):
        q = jnp.asarray(loss_fn.quantiles, dtype=pred.dtype)          # [Q]
        err = target_z[..., None] - pred                               # [B, h, Q]
        ql = jnp.maximum(q * err, (q - 1.0) * err)
        # Sum over quantiles and average over valid positions (mirrors
        # neuralforecast's MQLoss, whose per-quantile 1/len normalization is a no-op).
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


def train_on_windows(net, y_windows, target_mask, *, h, input_size, max_steps,
                     windows_batch_size, lr, seed, loss_fn, scaler,
                     futr_windows=None):
    """Train ``net`` in place on prebuilt window arrays via one ``nnx.scan``.

    Accepts pooled windows from any number of series (cross-learning); batch
    sampling is uniform over the pooled set — with replacement when there are
    fewer windows than ``windows_batch_size``, without replacement otherwise.
    Returns per-step losses.
    """
    n = y_windows.shape[0]
    step_keys = jax.random.split(jax.random.PRNGKey(seed), max_steps)
    if n < windows_batch_size:                          # fewer windows than batch: with replacement
        sample = lambda k: jax.random.choice(k, n, shape=(windows_batch_size,), replace=True)
    else:                                               # enough windows: without replacement
        sample = lambda k: jax.random.permutation(k, n)[:windows_batch_size]
    batch_idx = jax.vmap(sample)(step_keys)             # [max_steps, B]
    optimizer = nnx.Optimizer(net, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, idx):
        net, opt = carry
        yb = y_windows[idx]
        mb = target_mask[idx]
        fb = futr_windows[idx] if futr_windows is not None else None
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, yb, mb, h=h, input_size=input_size, scaler=scaler,
                                   loss_fn=loss_fn, futr_windows=fb))(net)
        opt.update(grads)
        return (net, opt), loss

    _, losses = step((net, optimizer), batch_idx)
    return _finite_or_raise(losses)


def train(net, y, *, h, input_size, max_steps, windows_batch_size, lr, seed, loss_fn, scaler,
          futr_exog=None):
    """Build windows from a single series and train (see ``train_on_windows``)."""
    y_windows, target_mask = build_windows(y, input_size, h)
    n = y_windows.shape[0]
    futr_w = build_exog_windows(futr_exog, input_size, h, n, "full") if futr_exog is not None else None
    return train_on_windows(
        net, y_windows, target_mask, h=h, input_size=input_size, max_steps=max_steps,
        windows_batch_size=windows_batch_size, lr=lr, seed=seed, loss_fn=loss_fn,
        scaler=scaler, futr_windows=futr_w,
    )


@nnx.jit
def _forward_det(net, insample_z, futr_z):
    """Inference forward. The MLP forward is deterministic — jit for dispatch speed."""
    return net(insample_z, futr_exog=futr_z)


def predict_step(net, y_context, *, h, input_size, scaler, futr_full=None):
    """Forecast next ``h`` steps from per-series contexts, in the **original**
    scale (point/quantile heads only — distribution heads go through
    ``predict_params``).

    ``y_context`` is ``[L]`` (one series) or ``[B, L]`` (a batch of series
    tails); returns ``[h, multiplier]`` / ``[B, h, multiplier]`` accordingly.
    ``futr_full`` (``[input_size+h, F]``, history + horizon) is shared across a
    batch of contexts.
    """
    single = y_context.ndim == 1
    ctx = y_context[None, :] if single else y_context   # [B, L]
    shift, scale = scaler.stats(ctx, axis=1)            # [B, 1]
    insample_z = scaler.transform(ctx, shift, scale)[..., None]        # [B, L, 1]
    futr_z = None
    if futr_full is not None:
        futr_z = _scale_exog(futr_full[None], scaler, stats_len=input_size)
        futr_z = jnp.broadcast_to(futr_z, (ctx.shape[0],) + futr_z.shape[1:])
    pred_z = _forward_det(net, insample_z, futr_z)      # [B, h, mult]
    out = scaler.inverse(pred_z, shift[..., None], scale[..., None])
    return out[0] if single else out


def predict_params(net, y_context, *, input_size, scaler, loss_fn, futr_full=None):
    """Distribution parameters for the next ``h`` steps, in the ORIGINAL scale.

    ``y_context`` is ``[L]`` (one series) or ``[B, L]`` (a batch of contexts —
    per-series tails); returns the loss's decoupled parameter tuple with arrays
    ``[h, K]`` / ``[B, h, K]`` accordingly. ``futr_full`` (``[input_size+h, F]``,
    history + horizon) is shared across a batch of contexts.
    """
    single = y_context.ndim == 1
    ctx = y_context[None, :] if single else y_context   # [B, L]
    shift, scale = scaler.stats(ctx, axis=1)            # [B, 1]
    insample_z = scaler.transform(ctx, shift, scale)[..., None]
    futr_z = None
    if futr_full is not None:
        futr_z = _scale_exog(futr_full[None], scaler, stats_len=input_size)
        futr_z = jnp.broadcast_to(futr_z, (ctx.shape[0],) + futr_z.shape[1:])
    raw = _forward_det(net, insample_z, futr_z)         # [B, h, mult]
    distr_args = loss_fn.domain_map(raw)
    distr_args = loss_fn.scale_decouple(distr_args, loc=shift[..., None],
                                        scale=scale[..., None])
    if single:
        distr_args = tuple(a[0] for a in distr_args)
    return distr_args
