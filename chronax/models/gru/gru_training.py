"""Window construction and JIT-compiled training/predict steps for the GRU."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler, Scaler


def build_windows(y: jnp.ndarray, input_size: int, h: int) -> jnp.ndarray:
    """Return [n_windows, input_size+h] rolling windows; step=1."""
    window_size = input_size + h
    n = y.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError(
            f"Series length {y.shape[0]} too short for input_size={input_size}, h={h}"
        )
    idx = jnp.arange(window_size)[None, :] + jnp.arange(n)[:, None]
    return y[idx]


def scaled_forward_loss(
    model: GRUNet,
    windows: jnp.ndarray,
    *,
    h: int,
    input_size: int,
    scaler: Scaler,
) -> jnp.ndarray:
    """MAE in scaled space, mirrors Nixtla. windows: [B, input_size+h] -> scalar."""
    insample = windows[:, :input_size]
    target = windows[:, input_size:]
    shift, scale = scaler.stats(insample, axis=1)
    insample_z = scaler.transform(insample, shift, scale)
    target_z = scaler.transform(target, shift, scale)
    pred = model(insample_z[..., None], deterministic=False)
    return jnp.mean(jnp.abs(pred[..., 0] - target_z))


@nnx.jit
def _jit_forward_deterministic(model: GRUNet, x: jnp.ndarray) -> jnp.ndarray:
    """Module-level JIT'd forward in inference mode. Cached across predict_step calls."""
    return model(x, deterministic=True)


def train(
    model: GRUNet,
    y: jnp.ndarray,
    *,
    h: int,
    input_size: int,
    max_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    scaler: Scaler | None = None,
) -> jnp.ndarray:
    """Train `model` in place. Returns array of per-step training losses.

    The train_step closure is JIT-compiled once and reused across
    `max_steps` — no per-step retracing.
    """
    if scaler is None:
        scaler = RobustScaler()
    windows = build_windows(y, input_size, h)
    n_windows = windows.shape[0]
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(
            lambda m: scaled_forward_loss(
                m, batch, h=h, input_size=input_size, scaler=scaler
            )
        )(model)
        optimizer.update(grads)
        return loss

    key = jax.random.PRNGKey(seed)
    losses = []
    for _ in range(max_steps):
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, n_windows, shape=(batch_size,), replace=True)
        losses.append(float(train_step(model, optimizer, windows[idx])))
    return jnp.asarray(losses)


def predict_step(
    model: GRUNet,
    y: jnp.ndarray,
    *,
    h: int,
    input_size: int,
    scaler: Scaler,
) -> jnp.ndarray:
    """Forecast next h steps from final input_size of y. Returns shape (h,).

    Delegates to module-level `_jit_forward_deterministic` so repeated calls
    on the same model do not re-trace.
    """
    insample = y[-input_size:][None, :]                 # [1, L]
    shift, scale = scaler.stats(insample, axis=1)       # [1, 1]
    x_z = scaler.transform(insample, shift, scale)[..., None]  # [1, L, 1]
    pred_z = _jit_forward_deterministic(model, x_z)            # [1, h, 1]
    return scaler.inverse(pred_z[..., 0], shift, scale)[0]
