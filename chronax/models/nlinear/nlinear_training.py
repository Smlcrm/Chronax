"""Window construction and JIT-compiled training/predict steps for NLinear."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chronax.models.nlinear.nlinear_losses import LossFn, mae
from chronax.models.nlinear.nlinear_module import NLinearNet
from chronax.models.nlinear.nlinear_scaler import Scaler


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rolling training windows with neuralforecast-style right-padding.

    Right-pads ``y`` with ``h`` zeros before unfolding (NF ``padder_train =
    ConstantPad1d((0, h))``), yielding ``len(y) - input_size`` windows of length
    ``input_size + h`` — including ~``h`` partial-horizon windows whose context
    reaches the end of the series. Returns ``(windows, mask)`` where ``mask`` is
    1 on real points and 0 on the padded tail, so the loss can drop padded
    horizon steps. The insample (first ``input_size``) of every window is fully
    real. Returns shape ``[n_windows, input_size+h]`` each.
    """
    window_size = input_size + h
    n_real = y.shape[0]
    n = n_real - input_size                              # NF keeps windows with >=1 valid outsample
    if n < 1:
        raise ValueError(f"Series length {n_real} too short for input_size={input_size}, h={h}")
    y_pad = jnp.concatenate([y, jnp.zeros((h,), y.dtype)])
    avail = jnp.concatenate([jnp.ones((n_real,), y.dtype), jnp.zeros((h,), y.dtype)])
    idx = jnp.arange(window_size)[None, :] + jnp.arange(n)[:, None]
    return y_pad[idx], avail[idx]


def scaled_forward_loss(model: NLinearNet, windows: jnp.ndarray, mask: jnp.ndarray, *, h: int,
                        input_size: int, scaler: Scaler, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Forward + masked point loss in SCALED space. windows/mask: [B, input_size+h] -> scalar.

    The insample is always real (padding is target-side only), so the scaler sees
    real values. Padded horizon steps are excluded via the outsample mask.
    """
    insample = windows[:, :input_size]
    target = windows[:, input_size:]
    out_mask = mask[:, input_size:]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)
    target_z = scaler.transform(target, shift, scale)
    pred = model(insample_z[..., None])
    return loss_fn(pred[..., 0], target_z, out_mask)


@nnx.jit
def _jit_forward(model: NLinearNet, x: jnp.ndarray) -> jnp.ndarray:
    return model(x)


def _sample_batch_idx(step_keys: jnp.ndarray, n_windows: int, windows_batch_size: int) -> jnp.ndarray:
    """Per-step window indices replicating NF's regimes: with-replacement when
    ``n_windows < windows_batch_size``, else a uniform random subset.

    The subset is the top-``windows_batch_size`` block of iid uniform keys —
    distribution-equivalent to NF's ``randperm(n)[:k]`` (by symmetry every
    k-subset is equally likely, and batch order is irrelevant to a mean-reduced
    loss). Selection runs via ``np.argpartition`` on the host when the keys are
    concrete: measured per fit at RoomTemperature scale (n≈7000, 5000 steps,
    the benchmark's only large-n dataset), sampling cost is 9.6s for a vmapped
    full permutation, 4.6s for ``lax.top_k``, 2.3s for jitted ``jnp.argpartition``
    and 0.4s for this hybrid — against ~0.5s for the ENTIRE training scan, since
    NLinear's per-step compute is one small matmul. Under a trace (e.g.
    ``BaseForecaster.conformity_scores``'s vmap, whose short CV windows normally
    hit the small-n branch anyway) it falls back to pure-JAX ``argpartition`` —
    both paths select the same index sets from the same uniforms.
    Returns ``[len(step_keys), windows_batch_size]`` int32.
    """
    if n_windows < windows_batch_size:
        def sample_one(k):
            return jax.random.choice(k, n_windows, shape=(windows_batch_size,), replace=True)
        return jax.vmap(sample_one)(step_keys)
    u = jax.vmap(lambda k: jax.random.uniform(k, (n_windows,)))(step_keys)
    cut = n_windows - windows_batch_size
    try:
        u_host = np.asarray(u)
    except jax.errors.TracerArrayConversionError:
        return jnp.argpartition(u, cut, axis=-1)[:, cut:]
    return jnp.asarray(np.argpartition(u_host, cut, axis=-1)[:, cut:])


def train(model: NLinearNet, y: jnp.ndarray, *, h: int, input_size: int, max_steps: int,
          windows_batch_size: int, lr: optax.ScalarOrSchedule, seed: int,
          scaler: Scaler, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Train `model` in place via a single nnx.scan. Returns per-step losses.

    Single nnx.scan (carry = (model, optimizer)) keeps train() vmap-traceable for
    BaseForecaster.conformity_scores. Window sampling replicates neuralforecast's
    regime-dependent scheme (with-replacement when n_windows < windows_batch_size,
    else a permutation). The scan iterates an int32 index tensor and gathers
    windows in-step (avoids materializing a large float32 batch tensor).
    """
    windows, mask = build_windows(y, input_size, h)
    n_windows = windows.shape[0]
    key = jax.random.PRNGKey(seed)
    step_keys = jax.random.split(key, max_steps)
    batch_idx = _sample_batch_idx(step_keys, n_windows, windows_batch_size)  # [max_steps, windows_batch_size] int32
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, idx):
        model, opt = carry
        batch_windows = windows[idx]
        batch_mask = mask[idx]
        loss, grads = nnx.value_and_grad(
            lambda m: scaled_forward_loss(m, batch_windows, batch_mask, h=h, input_size=input_size,
                                          scaler=scaler, loss_fn=loss_fn)
        )(model)
        opt.update(grads)
        return (model, opt), loss

    _, losses = step((model, optimizer), batch_idx)

    # Finite check only when concrete; under conformity_scores' vmap the loss is a
    # tracer and np.asarray raises, so return the traced array (NaNs surface as NaN
    # predictions downstream).
    try:
        losses_host = np.asarray(losses)
    except jax.errors.TracerArrayConversionError:
        return losses
    bad = ~np.isfinite(losses_host)
    if bad.any():
        first_bad = int(np.argmax(bad))
        raise RuntimeError(
            f"Non-finite loss ({losses_host[first_bad]}) at step {first_bad}. Training diverged. "
            f"Consider lowering `learning_rate`, reducing `windows_batch_size`, or using scaler='robust'."
        )
    return jnp.asarray(losses_host)


def predict_step(model: NLinearNet, y: jnp.ndarray, *, h: int, input_size: int, scaler: Scaler) -> jnp.ndarray:
    """Forecast next h steps from the final input_size of y, inverse-scaled. Returns (h,)."""
    insample = y[-input_size:][None, :]                  # [1, L]
    shift, scale = scaler.stats(insample, axis=1)
    x_z = scaler.transform(insample, shift, scale)[..., None]
    pred_z = _jit_forward(model, x_z)                    # [1, h, 1]
    return scaler.inverse(pred_z[..., 0], shift, scale)[0]
