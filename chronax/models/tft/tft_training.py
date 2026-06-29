"""Exog-aware window construction and JIT/scan training for TFT.

Windows are scaled per-window (robust on the insample target, per-channel on
continuous exog; static raw), the loss is computed in scaled space, and the whole
training loop is one ``nnx.scan`` so it stays ``vmap``-traceable for
``BaseForecaster.conformity_scores`` (mirrors the iTransformer trainer).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from chronax.models.tft.tft_losses import LossFn, MultiQuantileLoss
from chronax.models.tft.tft_module import TFTNet


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> jnp.ndarray:
    """Rolling windows ``[n_windows, input_size+h]`` of ``y`` (step 1)."""
    window = input_size + h
    n = y.shape[0] - window + 1
    if n <= 0:
        raise ValueError(f"Series length {y.shape[0]} too short for input_size={input_size}, h={h}.")
    idx = jnp.arange(window)[None, :] + jnp.arange(n)[:, None]
    return y[idx]


def build_exog_windows(arr: jnp.ndarray, input_size: int, h: int, n_windows: int, span: str) -> jnp.ndarray:
    """Rolling windows of an exog array ``[T, F]``.

    ``span="input"`` -> ``[n, input_size, F]`` (encoder window); ``span="full"`` ->
    ``[n, input_size+h, F]`` (future-known spanning input + horizon).
    """
    length = input_size if span == "input" else input_size + h
    idx = jnp.arange(length)[None, :] + jnp.arange(n_windows)[:, None]
    return arr[idx]


def _scale_exog(windows: jnp.ndarray, scaler) -> jnp.ndarray:
    """Per-channel per-window robust scaling of ``[B, T, F]`` exog."""
    shift, scale = scaler.stats(windows, axis=1)        # [B, 1, F]
    return scaler.transform(windows, shift, scale)


def forward_loss(net, y_windows, *, h, input_size, scaler, loss_fn,
                 hist_windows=None, futr_windows=None, stat=None, deterministic=False):
    """Scale, forward, and reduce a point/quantile loss in scaled space."""
    insample = y_windows[:, :input_size]                # [B, L]
    target = y_windows[:, input_size:]                  # [B, h]
    shift, scale = scaler.stats(insample, axis=1)       # [B, 1]
    insample_z = scaler.transform(insample, shift, scale)[..., None]   # [B, L, 1]
    target_z = scaler.transform(target, shift, scale)   # [B, h]
    hist_z = _scale_exog(hist_windows, scaler) if hist_windows is not None else None
    futr_z = _scale_exog(futr_windows, scaler) if futr_windows is not None else None
    pred = net(insample_z, hist_exog=hist_z, futr_exog=futr_z, stat_exog=stat,
               deterministic=deterministic)             # [B, h, mult]
    if isinstance(loss_fn, MultiQuantileLoss):
        return loss_fn(pred, target_z)
    return loss_fn(pred[..., 0], target_z)


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
          hist_exog=None, futr_exog=None, stat_exog=None):
    """Train ``net`` in place via one ``nnx.scan``. Returns per-step losses."""
    y_windows = build_windows(y, input_size, h)
    n = y_windows.shape[0]
    hist_w = build_exog_windows(hist_exog, input_size, h, n, "input") if hist_exog is not None else None
    futr_w = build_exog_windows(futr_exog, input_size, h, n, "full") if futr_exog is not None else None
    stat_w = jnp.broadcast_to(stat_exog[None, :], (n, stat_exog.shape[0])) if stat_exog is not None else None

    step_keys = jax.random.split(jax.random.PRNGKey(seed), max_steps)
    if n < windows_batch_size:                          # NF: torch.randint -> with replacement
        sample = lambda k: jax.random.choice(k, n, shape=(windows_batch_size,), replace=True)
    else:                                               # NF: torch.randperm[:B] -> without replacement
        sample = lambda k: jax.random.permutation(k, n)[:windows_batch_size]
    batch_idx = jax.vmap(sample)(step_keys)             # [max_steps, B]

    optimizer = nnx.Optimizer(net, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, idx):
        net, opt = carry
        yb = y_windows[idx]
        hb = hist_w[idx] if hist_w is not None else None
        fb = futr_w[idx] if futr_w is not None else None
        sb = stat_w[idx] if stat_w is not None else None
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, yb, h=h, input_size=input_size, scaler=scaler,
                                   loss_fn=loss_fn, hist_windows=hb, futr_windows=fb,
                                   stat=sb, deterministic=False)
        )(net)
        opt.update(grads)
        return (net, opt), loss

    _, losses = step((net, optimizer), batch_idx)
    return _finite_or_raise(losses)


@nnx.jit
def _forward_det(net, insample_z, hist_z, futr_z, stat_b):
    return net(insample_z, hist_exog=hist_z, futr_exog=futr_z, stat_exog=stat_b, deterministic=True)


def predict_step(net, y_context, *, h, input_size, scaler,
                 hist_context=None, futr_full=None, stat=None):
    """Forecast next ``h`` steps from the final ``input_size`` of the series.

    Returns ``[h, multiplier]`` in the **original** scale. ``futr_full`` is the
    ``[input_size+h, F]`` future-known window (history + horizon); ``hist_context``
    is the ``[input_size, H]`` encoder window; ``stat`` is ``[S]``.
    """
    insample = y_context[None, :]                       # [1, L]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)[..., None]   # [1, L, 1]
    hist_z = _scale_exog(hist_context[None], scaler) if hist_context is not None else None
    futr_z = _scale_exog(futr_full[None], scaler) if futr_full is not None else None
    stat_b = stat[None] if stat is not None else None
    pred_z = _forward_det(net, insample_z, hist_z, futr_z, stat_b)[0]  # [h, mult]
    return scaler.inverse(pred_z, shift[0, 0], scale[0, 0])
