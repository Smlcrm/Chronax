"""Window construction and JIT-compiled training/predict steps for the GRU."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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
    """MAE in scaled space. windows: [B, input_size+h] -> scalar."""
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
    """Train `model` in place. Returns per-step training losses.

    The training loop runs as a single `nnx.scan`, which makes the entire
    function `jax.vmap`-traceable. Mutable Flax NNX state (model parameters,
    optimizer momentum) flows through the scan as `nnx.Carry` — NNX handles
    the graph-vs-state separation internally.

    Per-step batches are pre-sampled outside the scan so the body is a pure
    function of `(model, optimizer, batch) -> mutated state, loss`. The
    non-finite-loss check is one host-side reduction over the loss array
    after the scan returns, rather than a per-step sync.
    """
    if scaler is None:
        scaler = RobustScaler()
    windows = build_windows(y, input_size, h)
    n_windows = windows.shape[0]

    # Pre-sample per-step batch indices. One choice() call → [max_steps, B];
    # the scan then iterates over the leading axis.
    key = jax.random.PRNGKey(seed)
    batch_idx = jax.random.choice(
        key, n_windows, shape=(max_steps, batch_size), replace=True
    )
    batches = windows[batch_idx]  # [max_steps, batch_size, L+h]

    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    # nnx.scan in flax 0.10.7 takes exactly one Carry slot. Pack model and
    # optimizer into a tuple carry; NNX recognizes the stateful members
    # inside and threads their graph state correctly.
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def step(carry, batch):
        model, opt = carry
        loss, grads = nnx.value_and_grad(
            lambda m: scaled_forward_loss(
                m, batch, h=h, input_size=input_size, scaler=scaler
            )
        )(model)
        opt.update(grads)
        return (model, opt), loss

    _, losses = step((model, optimizer), batches)  # shape: [max_steps]

    # Finite check: only when running concretely. When train() runs under a
    # higher-level jax trace (e.g. BaseForecaster.conformity_scores's vmap),
    # the loss array is a tracer and np.asarray would raise. In that path
    # we return the traced array and let the caller handle non-finite values
    # downstream (typically they'd surface as NaN in predictions).
    try:
        losses_host = np.asarray(losses)
    except jax.errors.TracerArrayConversionError:
        return losses
    bad = ~np.isfinite(losses_host)
    if bad.any():
        first_bad = int(np.argmax(bad))
        raise RuntimeError(
            f"Non-finite loss ({losses_host[first_bad]}) at step {first_bad}. "
            f"Training diverged. Consider lowering `learning_rate`, "
            f"reducing batch size, or checking the input series for extreme values."
        )

    return jnp.asarray(losses_host)


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
