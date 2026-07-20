"""Window construction and JIT/scan training for StemGNN.

Windows are scaled per-window (scaler stats on the insample target), the loss
is computed in scaled space, and the whole training loop is one ``nnx.scan`` so
it stays ``vmap``-traceable for ``BaseForecaster.conformity_scores`` (mirrors
the TCN/Informer trainers). Two RNG streams split off ``seed``: batch-index
keys (NF's regime-dependent window sampling) and per-step dropout keys for the
attention dropout (Informer's two-stream pattern) — shuffling batches never
perturbs which attention entries drop on a given step.

Unlike the older wrappers, StemGNN also mirrors NF's DEFAULT learning-rate
schedule: ``num_lr_decays=3`` -> torch ``StepLR(step_size=max_steps//3,
gamma=0.5)`` stepped per optimizer step (``_base_model.py`` default). The other
chronax neural ports silently train at constant LR against an NF that decays —
a filed template gap; StemGNN is the first port to close it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import chronax.models.stemgnn.stemgnn_losses as _losses
from chronax.models.stemgnn.stemgnn_losses import MultiQuantileLoss


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NF-parity rolling windows over ``y`` right-padded with ``h`` zeros.

    NF's ``padder_train = ConstantPad1d((0, h), 0)`` pads the training series
    before windowing, and the default availability threshold (0.0) keeps every
    window with >=1 valid target point, masking the padded tail out of the
    loss (the multivariate path verified identical to the univariate one at
    n_series=1). The partial windows put insample contexts ending at the very
    last observations into training — on trending series this is where the
    forecast-relevant regime lives (dropping them cost TCN/airline ~13 MAE,
    root-caused 2026-07-17 via lockstep replay).

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


# Elementwise forms of the registry point losses, for NF-parity masked reduction
# (NF losses compute sum(loss*mask)/sum(mask) — `_weighted_mean`). Keyed by the
# registry function OBJECTS so a user's custom callable that happens to share a
# name falls through to the custom branch instead of being shadowed.
_ELEMENTWISE = {
    _losses.mae: lambda e: jnp.abs(e),
    _losses.mse: lambda e: e * e,
    _losses.huber: lambda e: jnp.where(jnp.abs(e) <= 1.0, 0.5 * e * e, jnp.abs(e) - 0.5),
}


def _lr_schedule(lr: float, max_steps: int, num_lr_decays: int):
    """NF's default scheduler: torch ``StepLR(step_size=max(max_steps//num_lr_decays,
    1), gamma=0.5)`` stepped once per optimizer step.

    optax's update ``t`` (1-indexed) reads ``schedule(t-1)`` (the count
    pre-increment), and torch applies gamma when ``last_epoch`` reaches a
    multiple of ``step_size`` — so torch update ``t`` and optax count ``t-1``
    align at boundaries ``k*step_size``. For ``max_steps=1000, num_lr_decays=3``:
    updates 1-333 at ``lr``, 334-666 at ``lr/2``, 667-999 at ``lr/4``, update
    1000 at ``lr/8`` (boundaries {333, 666, 999}). ``num_lr_decays <= 0``
    disables the schedule (NF sets ``lr_decay_steps=10e7``).
    """
    if num_lr_decays is None or num_lr_decays <= 0:
        return lr
    d = max(max_steps // num_lr_decays, 1)
    boundaries = {k * d: 0.5 for k in range(1, (max_steps - 1) // d + 1)}
    if not boundaries:
        return lr
    return optax.piecewise_constant_schedule(lr, boundaries)


def forward_loss(net, y_windows, target_mask=None, *, h, input_size, scaler, loss_fn,
                 dropout_key=None, deterministic=True):
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
    pred = net(insample_z, dropout_key=dropout_key, deterministic=deterministic)  # [B, h, mult]
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


def train(net, y, *, h, input_size, max_steps, windows_batch_size, lr, num_lr_decays,
          seed, loss_fn, scaler):
    """Train ``net`` in place via one ``nnx.scan``. Returns per-step losses.

    Batch-index sampling follows NF's regime split: with replacement when the
    dataset has fewer windows than ``windows_batch_size`` (``torch.randint``),
    without replacement otherwise (``torch.randperm[:B]``). Adam runs on the
    NF-default StepLR schedule (``_lr_schedule``).
    """
    y_windows, target_mask = build_windows(y, input_size, h)
    n = y_windows.shape[0]
    batch_key, drop_key = jax.random.split(jax.random.PRNGKey(seed))
    step_keys = jax.random.split(batch_key, max_steps)
    if n < windows_batch_size:                          # NF: torch.randint -> with replacement
        sample = lambda k: jax.random.choice(k, n, shape=(windows_batch_size,), replace=True)
    else:                                               # NF: torch.randperm[:B] -> without replacement
        sample = lambda k: jax.random.permutation(k, n)[:windows_batch_size]
    batch_idx = jax.vmap(sample)(step_keys)             # [max_steps, B]
    drop_keys = jax.random.split(drop_key, max_steps)   # [max_steps, 2]
    optimizer = nnx.Optimizer(
        net, optax.adam(_lr_schedule(lr, max_steps, num_lr_decays)), wrt=nnx.Param
    )

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, xs):
        net, opt = carry
        idx, dkey = xs
        yb = y_windows[idx]
        mb = target_mask[idx]
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, yb, mb, h=h, input_size=input_size, scaler=scaler,
                                   loss_fn=loss_fn, dropout_key=dkey,
                                   deterministic=False))(net)
        opt.update(grads)
        return (net, opt), loss

    _, losses = step((net, optimizer), (batch_idx, drop_keys))
    return _finite_or_raise(losses)


@nnx.jit
def _forward_det(net, insample_z):
    """Inference forward: deterministic (dropout off, as under torch ``model.eval()``)."""
    return net(insample_z)


def predict_step(net, y_context, *, h, input_size, scaler):
    """Forecast next ``h`` steps from the final ``input_size`` of the series.

    Returns ``[h, multiplier]`` in the **original** scale.
    """
    insample = y_context[None, :]                       # [1, L]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)[..., None]   # [1, L, 1]
    pred_z = _forward_det(net, insample_z)[0]           # [h, mult]
    return scaler.inverse(pred_z, shift[0, 0], scale[0, 0])
