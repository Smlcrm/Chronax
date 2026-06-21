"""Window construction and JIT-compiled training/predict steps for KAN."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chronax.models.kan.kan_losses import LossFn, mae
from chronax.models.kan.kan_module import KANNet
from chronax.models.kan.kan_scaler import Scaler


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> jnp.ndarray:
    """Return [n_windows, input_size+h] rolling windows; step=1."""
    window_size = input_size + h
    n = y.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError(f"Series length {y.shape[0]} too short for input_size={input_size}, h={h}")
    idx = jnp.arange(window_size)[None, :] + jnp.arange(n)[:, None]
    return y[idx]


def scaled_forward_loss(model: KANNet, windows: jnp.ndarray, *, h: int, input_size: int,
                        scaler: Scaler, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Forward + point loss in SCALED space. windows: [B, input_size+h] -> scalar."""
    insample = windows[:, :input_size]
    target = windows[:, input_size:]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)
    target_z = scaler.transform(target, shift, scale)
    pred = model(insample_z[..., None])
    return loss_fn(pred[..., 0], target_z)


@nnx.jit
def _jit_forward(model: KANNet, x: jnp.ndarray) -> jnp.ndarray:
    return model(x)


def train(model: KANNet, y: jnp.ndarray, *, h: int, input_size: int, max_steps: int,
          windows_batch_size: int, lr: optax.ScalarOrSchedule, seed: int,
          scaler: Scaler, loss_fn: LossFn = mae) -> jnp.ndarray:
    """Train `model` in place via a single nnx.scan. Returns per-step losses.

    Single nnx.scan (carry = (model, optimizer)) keeps train() vmap-traceable for
    BaseForecaster.conformity_scores. Window sampling replicates neuralforecast's
    regime-dependent scheme (with-replacement when n_windows < windows_batch_size,
    else a permutation). The scan iterates an int32 index tensor and gathers
    windows in-step (avoids materializing a large float32 batch tensor).
    """
    windows = build_windows(y, input_size, h)
    n_windows = windows.shape[0]
    key = jax.random.PRNGKey(seed)
    step_keys = jax.random.split(key, max_steps)

    if n_windows < windows_batch_size:
        def sample_one(k):
            return jax.random.choice(k, n_windows, shape=(windows_batch_size,), replace=True)
    else:
        def sample_one(k):
            return jax.random.permutation(k, n_windows)[:windows_batch_size]

    batch_idx = jax.vmap(sample_one)(step_keys)          # [max_steps, windows_batch_size] int32
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, idx):
        model, opt = carry
        batch_windows = windows[idx]
        loss, grads = nnx.value_and_grad(
            lambda m: scaled_forward_loss(m, batch_windows, h=h, input_size=input_size,
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


def predict_step(model: KANNet, y: jnp.ndarray, *, h: int, input_size: int, scaler: Scaler) -> jnp.ndarray:
    """Forecast next h steps from the final input_size of y, inverse-scaled. Returns (h,)."""
    insample = y[-input_size:][None, :]                  # [1, L]
    shift, scale = scaler.stats(insample, axis=1)
    x_z = scaler.transform(insample, shift, scale)[..., None]
    pred_z = _jit_forward(model, x_z)                    # [1, h, 1]
    return scaler.inverse(pred_z[..., 0], shift, scale)[0]
