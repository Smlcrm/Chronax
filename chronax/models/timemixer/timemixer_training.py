"""Window construction and JIT/scan training for TimeMixer.

Windows are h-padded partial windows with a target mask (every window keeps at
least one real target and the newest observations appear as training contexts;
the zero-padded tail is masked out of the loss). The loss is computed in
ORIGINAL scale — normalization happens inside the network (per-scale RevIN),
matching the reference's identity-scaler arrangement. The whole training loop
is one ``nnx.scan`` so it stays ``vmap``-traceable for
``BaseForecaster.conformity_scores``; dropout keys advance through the scan
carry via the model's ``nnx.Rngs``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

import chronax.models.timemixer.timemixer_losses as _losses
from chronax.models.timemixer.timemixer_losses import LossFn
from chronax.models.timemixer.timemixer_module import TimeMixerNet


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rolling windows over ``y``, right-padded with ``h`` zeros.

    Returns ``(windows [n, input_size+h], target_mask [n, h])`` with
    ``n = len(y) - input_size``; the mask is 1.0 at real target positions and
    0.0 in the zero-padded tail.
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


# Elementwise forms of the registry point losses, for masked reduction
# (sum(loss*mask)/sum(mask)). Keyed by the registry function OBJECTS so a
# custom callable that shares a name falls through to the custom branch.
_ELEMENTWISE = {
    _losses.mae: lambda e: jnp.abs(e),
    _losses.mse: lambda e: e * e,
    _losses.huber: lambda e: jnp.where(jnp.abs(e) <= 1.0, 0.5 * e * e, jnp.abs(e) - 0.5),
}


def forward_loss(model: TimeMixerNet, y_windows: jnp.ndarray,
                 target_mask: jnp.ndarray | None = None, *, h: int,
                 input_size: int, loss_fn: LossFn) -> jnp.ndarray:
    """Forward + masked point loss in ORIGINAL scale.

    ``y_windows [B, input_size+h]``; the univariate series carries a channel
    dim of 1 into the N-generic network.
    """
    insample = y_windows[:, :input_size][..., None]     # [B, L, 1]
    target = y_windows[:, input_size:]                  # [B, h]
    pred = model(insample, deterministic=False)[..., 0]  # [B, h]
    if target_mask is None:
        target_mask = jnp.ones_like(target)
    ew = _ELEMENTWISE.get(loss_fn)
    if ew is not None:
        return jnp.sum(ew(pred - target) * target_mask) / jnp.sum(target_mask)
    # Custom callable: substitute masked targets so padded positions are exact
    # zeros of the loss; the denominator stays the callable's own.
    return loss_fn(jnp.where(target_mask > 0, pred, target), target)


def _finite_or_raise(losses: jnp.ndarray) -> jnp.ndarray:
    """Divergence guard. No-ops under a higher trace (e.g. conformity_scores's
    vmap), where ``losses`` is a tracer and cannot be concretized."""
    if isinstance(losses, jax.core.Tracer):
        return losses
    if not bool(jnp.all(jnp.isfinite(losses))):
        i = int(jnp.argmax(~jnp.isfinite(losses)))
        raise RuntimeError(
            f"Non-finite loss at step {i}. Training diverged. Lower learning_rate "
            "or windows_batch_size, or check the series for extreme values."
        )
    return losses


def train(model: TimeMixerNet, y: jnp.ndarray, *, h: int, input_size: int,
          max_steps: int, windows_batch_size: int, lr, seed: int,
          loss_fn: LossFn) -> jnp.ndarray:
    """Train ``model`` in place via one ``nnx.scan``. Returns per-step losses.

    Window sampling replicates the reference's regime-dependent scheme: with
    fewer windows than ``windows_batch_size`` indices are drawn WITH
    replacement, otherwise a without-replacement permutation slice.
    """
    y_windows, target_mask = build_windows(y, input_size, h)
    n = y_windows.shape[0]
    step_keys = jax.random.split(jax.random.PRNGKey(seed), max_steps)
    if n < windows_batch_size:
        sample = lambda k: jax.random.choice(k, n, shape=(windows_batch_size,), replace=True)
    else:
        sample = lambda k: jax.random.permutation(k, n)[:windows_batch_size]
    batch_idx = jax.vmap(sample)(step_keys)             # [max_steps, B]
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, idx):
        model, opt = carry
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, y_windows[idx], target_mask[idx], h=h,
                                   input_size=input_size, loss_fn=loss_fn))(model)
        opt.update(grads)
        return (model, opt), loss

    _, losses = step((model, optimizer), batch_idx)
    return _finite_or_raise(losses)


@nnx.jit
def _forward_det(model: TimeMixerNet, x: jnp.ndarray) -> jnp.ndarray:
    """Inference-mode forward (dropout disabled). Cached across calls; the
    value-equal initializers keep same-config graphdefs cache-equal across
    refits."""
    return model(x, deterministic=True)


def predict_step(model: TimeMixerNet, y: jnp.ndarray, *, h: int, input_size: int) -> jnp.ndarray:
    """Forecast next ``h`` steps from the final ``input_size`` of ``y``.
    Returns ``[h, mult]`` in original scale (the net denormalizes internally)."""
    x = y[-input_size:][None, :, None]                  # [1, L, 1]
    pred = _forward_det(model, x)                       # [1, h, mult]
    return pred[0]
