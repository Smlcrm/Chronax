"""Window construction and JIT/scan training for NHITS.

Windows are scaled per-window (scaler stats on the insample target; per-channel
scaling on historical and future-known exog) and the whole training loop is one ``nnx.scan``
so it stays ``vmap``-traceable for ``BaseForecaster.conformity_scores``. Point
and multi-quantile losses are computed in scaled space; distribution losses
(``GMM``) are evaluated against the ORIGINAL-scale target with the predicted
parameters mapped out of scaled space via ``scale_decouple`` — the network
optimizes in the scaler's normalized range while the likelihood lives in data
units. At the default ``dropout_prob_theta=0`` the forward is fully
deterministic, so the only RNG stream is the batch-index sampling; with
dropout enabled the keys advance through the scan carry via the model's
``nnx.Rngs`` and inference runs with dropout disabled.

``num_lr_decays`` maps to a torch-style ``StepLR(step_size=max_steps //
num_lr_decays, gamma=0.5)`` stepped once per optimizer step (active at the
reference NHITS default of 3, unlike the constant-rate MLP).

``train`` = ``build_windows`` + ``train_on_windows``; the second half accepts
prebuilt (possibly pooled) window arrays so a multi-series caller can
cross-learn one net over many series' windows.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import functools

import optax
from flax import nnx

import chronax.models.nhits.nhits_losses as _losses
from chronax.models.nhits.nhits_losses import LossFn, MultiQuantileLoss
from chronax.models.nhits.nhits_module import NHITSNet


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


_EXOG_SCALE_FLOOR_FRAC = 0.01  # floor = frac * window range  =>  |scaled| <= 1/frac


def _scale_exog(windows: jnp.ndarray, scaler, stats_len: int | None = None) -> jnp.ndarray:
    """Per-channel per-window robust scaling of ``[B, T, F]`` exog.

    ``stats_len`` restricts the STATISTICS to the first ``stats_len`` positions
    (the insample span) while transforming the whole window, so exog stats never
    see the horizon slice. ``None`` keeps full-span stats (used only for
    insample-span windows).
    """
    stats_src = windows if stats_len is None else windows[:, :stats_len]
    shift, scale = scaler.stats(stats_src, axis=1)      # [B, 1, F]
    # A near-constant stats span collapses the robust scale to its epsilon, and
    # any regime-shifted value in the window then divides to a 1e5-scale input
    # (diurnal covariates: an all-night span before a daytime horizon). The
    # SPREAD-based floor bounds |scaled| <= range/(frac*range) = 1/frac by
    # construction (the shift lies inside [min, max]). frac sits ~15x below the
    # Gaussian scale/range ratio (~0.15 at these window lengths), so ordinary
    # covariates — including level-offset ones like temperature in Kelvin —
    # never trip it, while degenerate spans (scale/range ~ 1e-9) always do.
    # Covariate values are known inputs, so the bound may read the full window;
    # the CENTERING stays insample-span.
    rng_full = (jnp.max(windows, axis=1, keepdims=True)
                - jnp.min(windows, axis=1, keepdims=True))
    return scaler.transform(windows, shift,
                            jnp.maximum(scale, _EXOG_SCALE_FLOOR_FRAC * rng_full))


def _lr_schedule(lr: float, max_steps: int, num_lr_decays: int):
    """Torch-style ``StepLR(step_size=max(max_steps // num_lr_decays, 1),
    gamma=0.5)`` stepped once per optimizer step.

    optax's update ``t`` (1-indexed) reads ``schedule(t-1)`` (the count is read
    pre-increment), and torch applies gamma when ``last_epoch`` reaches a
    multiple of ``step_size``, so torch update ``t`` and optax count ``t-1``
    align at boundaries ``k*step_size``. For ``max_steps=1000, num_lr_decays=3``:
    updates 1-333 at ``lr``, 334-666 at ``lr/2``, 667-999 at ``lr/4``, update
    1000 at ``lr/8`` (boundaries {333, 666, 999}). ``num_lr_decays <= 0``
    disables the schedule.
    """
    if num_lr_decays is None or num_lr_decays <= 0:
        return lr
    d = max(max_steps // num_lr_decays, 1)
    boundaries = {k * d: 0.5 for k in range(1, (max_steps - 1) // d + 1)}
    if not boundaries:
        return lr
    return optax.piecewise_constant_schedule(lr, boundaries)


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
                 hist_windows=None, futr_windows=None, deterministic=True):
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
    # Historical exog spans only the input (stats read the whole window);
    # future-known exog spans input+horizon (stats restricted to the input span).
    hist_z = (_scale_exog(hist_windows, scaler, stats_len=None)
              if hist_windows is not None else None)
    futr_z = (_scale_exog(futr_windows, scaler, stats_len=input_size)
              if futr_windows is not None else None)
    pred = net(insample_z, hist_exog=hist_z, futr_exog=futr_z,
               deterministic=deterministic)             # [B, h, mult]
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
                     windows_batch_size, lr, num_lr_decays, seed, loss_fn, scaler,
                     hist_windows=None, futr_windows=None):
    """Train ``net`` in place on prebuilt window arrays via one ``nnx.scan``.

    Accepts pooled windows from any number of series (cross-learning); batch
    sampling is uniform over the pooled set — with replacement when there are
    fewer windows than ``windows_batch_size``, without replacement otherwise
    (the reference's two-stage series-then-window sampling differs from this
    uniform pooled draw only when there are more than 32 series). Returns
    per-step losses.
    """
    n = y_windows.shape[0]
    step_keys = jax.random.split(jax.random.PRNGKey(seed), max_steps)
    if n < windows_batch_size:                          # fewer windows than batch: with replacement
        sample = lambda k: jax.random.choice(k, n, shape=(windows_batch_size,), replace=True)
    else:                                               # enough windows: without replacement
        sample = lambda k: jax.random.permutation(k, n)[:windows_batch_size]
    batch_idx = jax.vmap(sample)(step_keys)             # [max_steps, B]
    optimizer = nnx.Optimizer(net, _adam_steplr(lr, max_steps, num_lr_decays), wrt=nnx.Param)
    losses = _train_scan(net, optimizer, y_windows, target_mask, batch_idx,
                         hist_windows, futr_windows,
                         h=h, input_size=input_size, scaler=scaler, loss_fn=loss_fn)
    return _finite_or_raise(losses)


@functools.lru_cache(maxsize=None)
def _adam_steplr(lr: float, max_steps: int, num_lr_decays: int):
    """One optax transform per (lr, max_steps, num_lr_decays). The optimizer's
    graphdef embeds the transform object, so a fresh ``optax.adam`` per fit
    would make same-config optimizers unequal and defeat the cross-fit
    ``_train_scan`` cache."""
    return optax.adam(_lr_schedule(lr, max_steps, num_lr_decays))


@functools.partial(nnx.jit, static_argnames=("h", "input_size", "scaler", "loss_fn"))
def _train_scan(net, optimizer, y_windows, target_mask, batch_idx, hist_windows, futr_windows,
                *, h, input_size, scaler, loss_fn):
    """The whole training loop as one cached program.

    Module-level so the traced/compiled program is reused across ``fit()`` calls:
    the jit cache keys on the net/optimizer graphdefs (value-``__eq__``
    initializers, the ``_adam_steplr`` and scaler singletons make same-config
    instances equal), the static config args, and operand shapes -- never on
    data. A per-call scan closure would retrace and recompile every fit, because
    pjit caches on the callable's identity.
    """

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, idx):
        net, opt = carry
        yb = y_windows[idx]
        mb = target_mask[idx]
        hb = hist_windows[idx] if hist_windows is not None else None
        fb = futr_windows[idx] if futr_windows is not None else None
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, yb, mb, h=h, input_size=input_size, scaler=scaler,
                                   loss_fn=loss_fn, hist_windows=hb, futr_windows=fb,
                                   deterministic=False))(net)
        opt.update(grads)
        return (net, opt), loss

    _, losses = step((net, optimizer), batch_idx)
    return losses


def train(net, y, *, h, input_size, max_steps, windows_batch_size, lr, num_lr_decays,
          seed, loss_fn, scaler, hist_exog=None, futr_exog=None):
    """Build windows from a single series and train (see ``train_on_windows``)."""
    y_windows, target_mask = build_windows(y, input_size, h)
    n = y_windows.shape[0]
    hist_w = build_exog_windows(hist_exog, input_size, h, n, "input") if hist_exog is not None else None
    futr_w = build_exog_windows(futr_exog, input_size, h, n, "full") if futr_exog is not None else None
    return train_on_windows(
        net, y_windows, target_mask, h=h, input_size=input_size, max_steps=max_steps,
        windows_batch_size=windows_batch_size, lr=lr, num_lr_decays=num_lr_decays,
        seed=seed, loss_fn=loss_fn, scaler=scaler, hist_windows=hist_w, futr_windows=futr_w,
    )


@nnx.jit
def _forward_det(net, insample_z, hist_z, futr_z):
    """Inference-mode forward (dropout disabled). Cached across calls; the
    value-equal initializers keep same-config graphdefs cache-equal across
    refits."""
    return net(insample_z, hist_exog=hist_z, futr_exog=futr_z, deterministic=True)


def _broadcast_exog(full, ctx_rows, *, input_size, scaler, stats_len):
    """Scale a single shared exog window ``[span, F]`` and broadcast to ``ctx_rows``
    contexts. ``stats_len`` restricts the scaler stats to the insample span
    (``None`` = whole window, used for input-span historical exog)."""
    z = _scale_exog(full[None], scaler, stats_len=stats_len)
    return jnp.broadcast_to(z, (ctx_rows,) + z.shape[1:])


def predict_step(net, y_context, *, h, input_size, scaler, hist_full=None, futr_full=None):
    """Forecast next ``h`` steps from per-series contexts, in the **original**
    scale (point/quantile heads only — distribution heads go through
    ``predict_params``).

    ``y_context`` is ``[L]`` (one series) or ``[B, L]`` (a batch of series
    tails); returns ``[h, multiplier]`` / ``[B, h, multiplier]`` accordingly.
    ``hist_full`` (``[input_size, F]``) and ``futr_full`` (``[input_size+h, F]``)
    are each shared across a batch of contexts.
    """
    single = y_context.ndim == 1
    ctx = y_context[None, :] if single else y_context   # [B, L]
    shift, scale = scaler.stats(ctx, axis=1)            # [B, 1]
    insample_z = scaler.transform(ctx, shift, scale)[..., None]        # [B, L, 1]
    hist_z = None if hist_full is None else _broadcast_exog(
        hist_full, ctx.shape[0], input_size=input_size, scaler=scaler, stats_len=None)
    futr_z = None if futr_full is None else _broadcast_exog(
        futr_full, ctx.shape[0], input_size=input_size, scaler=scaler, stats_len=input_size)
    pred_z = _forward_det(net, insample_z, hist_z, futr_z)   # [B, h, mult]
    out = scaler.inverse(pred_z, shift[..., None], scale[..., None])
    return out[0] if single else out


def predict_params(net, y_context, *, input_size, scaler, loss_fn, hist_full=None, futr_full=None):
    """Distribution parameters for the next ``h`` steps, in the ORIGINAL scale.

    ``y_context`` is ``[L]`` (one series) or ``[B, L]`` (a batch of contexts —
    per-series tails); returns the loss's decoupled parameter tuple with arrays
    ``[h, K]`` / ``[B, h, K]`` accordingly. ``hist_full`` (``[input_size, F]``)
    and ``futr_full`` (``[input_size+h, F]``) are each shared across a batch of
    contexts.
    """
    single = y_context.ndim == 1
    ctx = y_context[None, :] if single else y_context   # [B, L]
    shift, scale = scaler.stats(ctx, axis=1)            # [B, 1]
    insample_z = scaler.transform(ctx, shift, scale)[..., None]
    hist_z = None if hist_full is None else _broadcast_exog(
        hist_full, ctx.shape[0], input_size=input_size, scaler=scaler, stats_len=None)
    futr_z = None if futr_full is None else _broadcast_exog(
        futr_full, ctx.shape[0], input_size=input_size, scaler=scaler, stats_len=input_size)
    raw = _forward_det(net, insample_z, hist_z, futr_z)     # [B, h, mult]
    distr_args = loss_fn.domain_map(raw)
    distr_args = loss_fn.scale_decouple(distr_args, loc=shift[..., None],
                                        scale=scale[..., None])
    if single:
        distr_args = tuple(a[0] for a in distr_args)
    return distr_args
