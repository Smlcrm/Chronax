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

from chronax.models.informer.informer_losses import LossFn, MultiQuantileLoss
from chronax.models.informer.informer_module import InformerNet


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
                 futr_windows=None, sample_key, deterministic=False):
    """Scale, forward, and reduce a point/quantile loss in scaled space."""
    insample = y_windows[:, :input_size]                # [B, L]
    target = y_windows[:, input_size:]                  # [B, h]
    shift, scale = scaler.stats(insample, axis=1)       # [B, 1]
    insample_z = scaler.transform(insample, shift, scale)[..., None]   # [B, L, 1]
    target_z = scaler.transform(target, shift, scale)   # [B, h]
    futr_z = _scale_exog(futr_windows, scaler) if futr_windows is not None else None
    pred = net(insample_z, futr_exog=futr_z, sample_key=sample_key,
               deterministic=deterministic, use_running_average=deterministic)   # [B, h, mult]
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
          futr_exog=None):
    """Train ``net`` in place via one ``nnx.scan``. Returns per-step losses.

    Two independent RNG streams are split off ``seed``: batch-index keys (NF's
    regime-dependent window sampling, below) and ProbSparse ``attn_keys`` (one
    per scan step, threaded through to every attention site inside
    ``InformerNet``) -- so shuffling the training batches never perturbs which
    ProbSparse queries get sampled for a given step.
    """
    y_windows = build_windows(y, input_size, h)
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
        fb = futr_w[idx] if futr_w is not None else None
        loss, grads = nnx.value_and_grad(
            lambda m: forward_loss(m, yb, h=h, input_size=input_size, scaler=scaler,
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
    futr_z = _scale_exog(futr_full[None], scaler) if futr_full is not None else None
    pred_z = _forward_det(net, insample_z, futr_z)[0]  # [h, mult]
    return scaler.inverse(pred_z, shift[0, 0], scale[0, 0])
