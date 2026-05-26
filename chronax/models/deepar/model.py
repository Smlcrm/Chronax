"""DeepAR Encoder-Decoder model (FLAX/JAX/OPTAX implementation).

Core model definition with LSTM encoder-decoder architecture for
probabilistic time series forecasting. NumPy-free — all arrays are
jnp.ndarray throughout.
"""

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import flax.linen as nn
import optax


# -------------------------
# Utility
# -------------------------


def nll_gauss(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Gaussian negative log-likelihood with numerical stability.

    Args:
        y: Target values, shape (...)
        mu: Mean predictions, shape (...)
        sigma: Standard deviation predictions, shape (...)

    Returns:
        Per-element NLL, shape (...)
    """
    sigma = jnp.clip(sigma, 1e-6, 1e6)
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2


# -------------------------
# Model
# -------------------------


class DeepAR_EncDec(nn.Module):
    """DeepAR Encoder-Decoder in Flax/JAX.

    Architecture:
    - Encoder LSTM: consumes y_{1:T} (+ optional hist_exog / stat_exog) →
      final hidden state (h_T, c_T).
    - Decoder LSTM: generates Gaussian (μ, σ) autoregressively using
      teacher forcing during training; Monte Carlo rollout at inference.

    Args:
        hidden: Hidden dimension for both encoder and decoder LSTMs.
        dropout_rate: Dropout probability (applied in training mode).
        min_sigma: Minimum σ floor added after softplus for stability.
    """

    hidden: int = 64
    dropout_rate: float = 0.1
    min_sigma: float = 0.02

    def setup(self):
        self.proj = nn.Dense(self.hidden)

        # Shared LSTM cell unrolled over time via nn.scan.
        # variable_broadcast keeps params fixed across time steps.
        self.lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(features=self.hidden)

        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.head_mu = nn.Dense(1)
        self.head_sigma = nn.Dense(1)

    # ---------- Encoder ----------

    def encode(
        self,
        y_hist: jnp.ndarray,
        futr_exog: jnp.ndarray = None,
        x_static: jnp.ndarray = None,
    ):
        """Encode history into initial decoder hidden state.

        Args:
            y_hist: Historical targets [T].
            futr_exog: Future exogenous features for the history window [T, F] or None.
            x_static: Static features [S] or None.

        Returns:
            (h_T, c_T): LSTM hidden/cell states, each [1, hidden].
        """
        T = y_hist.shape[0]

        parts = [y_hist[:, None]]  # [T, 1]
        if futr_exog is not None:
            parts.append(futr_exog)  # [T, F]
        if x_static is not None:
            parts.append(jnp.tile(x_static[None, :], (T, 1)))  # [T, S]

        x_enc = jnp.concatenate(parts, axis=-1)[None, :, :]  # [1, T, D]

        h0 = jnp.zeros((1, self.hidden), dtype=jnp.float32)
        c0 = jnp.zeros((1, self.hidden), dtype=jnp.float32)
        x_proj = self.proj(x_enc)
        (hT, cT), _ = self.lstm((h0, c0), x_proj)
        return hT, cT

    # ---------- Decoder — single step ----------

    def one_step(
        self,
        y_prev_scalar: jnp.ndarray,
        x_f_step: jnp.ndarray = None,
        x_static: jnp.ndarray = None,
        h: jnp.ndarray = None,
        c: jnp.ndarray = None,
        deterministic: bool = True,
    ):
        """One autoregressive decoder step.

        Args:
            y_prev_scalar: Previous target value (scalar).
            x_f_step: Future exogenous at this step [F] or None.
            x_static: Static features [S] or None.
            h: Current hidden state [1, hidden] or None (zeros).
            c: Current cell state [1, hidden] or None (zeros).
            deterministic: If True, dropout is disabled.

        Returns:
            (mu, sigma, h_new, c_new)
        """
        if h is None:
            h = jnp.zeros((1, self.hidden))
        if c is None:
            c = jnp.zeros((1, self.hidden))

        parts = [y_prev_scalar[None]]
        if x_f_step is not None:
            parts.append(x_f_step)
        if x_static is not None:
            parts.append(x_static)

        x_vec = jnp.concatenate(parts, axis=0)[None, None, :]  # [1, 1, D]
        x_proj = self.proj(x_vec)

        (h_new, c_new), hs = self.lstm((h, c), x_proj)

        z = hs[:, -1, :]
        z = self.dropout(z, deterministic=deterministic)

        mu = self.head_mu(z)[..., 0]
        sraw = self.head_sigma(z)[..., 0]
        sigma = nn.softplus(sraw) + jnp.maximum(self.min_sigma, 1e-3)

        return mu[0], sigma[0], h_new, c_new

    # ---------- Training roll — teacher forcing ----------

    def __call__(
        self,
        y_seq: jnp.ndarray,
        futr_exog: jnp.ndarray = None,
        x_static: jnp.ndarray = None,
        training: bool = True,
    ):
        """Unroll decoder with teacher forcing over the full input window.

        Args:
            y_seq: Target values to feed into the model [L]. (y_{t-1})
            futr_exog: Future exogenous aligned with targets [L, F] or None.
            x_static: Static features [S] or None.
            training: If True, apply dropout.

        Returns:
            (mu, sigma): Gaussian parameters [L].
        """
        L = y_seq.shape[0]

        # Static features tiled over steps
        xs_rep = (
            jnp.zeros((L, 0), dtype=jnp.float32)
            if x_static is None
            else jnp.tile(x_static[None, :], (L, 1))
        )

        y_t = y_seq[:, None]  # inputs [L, 1]

        # Combine all inputs at time t
        parts = [y_t]
        if futr_exog is not None:
            parts.append(futr_exog)
        parts.append(xs_rep)
        
        dec_inputs = jnp.concatenate(parts, axis=-1)
        dec_inputs = dec_inputs[None, :, :]  # [1, L, D]

        x_proj = self.proj(dec_inputs)
        z0 = self.dropout(x_proj, deterministic=not training)

        h0 = jnp.zeros((1, self.hidden), dtype=jnp.float32)
        c0 = jnp.zeros((1, self.hidden), dtype=jnp.float32)

        (_, _), hs = self.lstm((h0, c0), z0)
        hs = hs.squeeze(0)  # [L, hidden]
        hs = self.dropout(hs, deterministic=not training)

        mu = self.head_mu(hs)[..., 0]   # [L]
        sraw = self.head_sigma(hs)[..., 0]
        sigma = nn.softplus(sraw) + jnp.maximum(self.min_sigma, 1e-3)

        return mu, sigma


# -------------------------
# Training utilities
# -------------------------


def make_trainer(model: DeepAR_EncDec, lr: float = 1e-3, weight_decay: float = 1e-5):
    """Build an Optax optimizer and a JIT-compiled training step.

    Args:
        model: DeepAR_EncDec instance.
        lr: Learning rate.
        weight_decay: AdamW weight-decay coefficient.

    Returns:
        (tx, step_fn) where step_fn(params, opt_state, y, xf, xs, key)
        returns (params, opt_state, loss).
    """
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(params, y_hist, futr_exog, x_static, key):
        # We assume y_hist here is the full L+h sequence and we predict the next step.
        mu, sigma = model.apply(
            params,
            y_hist[:-1],
            futr_exog[1:] if futr_exog is not None else None,
            x_static,
            True,
            rngs={"dropout": key},
        )
        y_target = y_hist[1:]
        return jnp.mean(nll_gauss(y_target, mu, sigma))

    @jit
    def step(params, opt_state, y_hist, futr_exog, x_static, key):
        loss, grads = value_and_grad(loss_fn)(params, y_hist, futr_exog, x_static, key)
        updates, opt_state_new = tx.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return tx, step


def train_model(
    y_hist: jnp.ndarray,
    x_f_all: jnp.ndarray = None,
    x_static: jnp.ndarray = None,
    hidden: int = 64,
    lr: float = 1e-3,
    steps: int = 800,
    dropout: float = 0.1,
    min_sigma: float = 0.02,
    verbose: bool = True,
):
    """Convenience training loop for a single time series.

    Args:
        y_hist: Target series [T].
        x_f_all: Future exogenous features [T, F] or None.
        x_static: Static features [S] or None.
        hidden: LSTM hidden size.
        lr: Learning rate.
        steps: Training steps.
        dropout: Dropout rate.
        min_sigma: Minimum σ floor.
        verbose: Print loss every 100 steps.

    Returns:
        (model, params, losses)
    """
    model = DeepAR_EncDec(hidden=hidden, dropout_rate=dropout, min_sigma=min_sigma)

    L = y_hist.shape[0]
    d_f = 0 if x_f_all is None else int(x_f_all.shape[1])
    d_s = 0 if x_static is None else int(x_static.shape[0])

    key = random.PRNGKey(1)
    dummy_xf = None if x_f_all is None else jnp.zeros((L, d_f), dtype=jnp.float32)
    dummy_xs = None if x_static is None else jnp.zeros((d_s,), dtype=jnp.float32)

    params = model.init(
        {"params": key, "dropout": key},
        jnp.array(y_hist[:-1]),
        dummy_xf[1:] if dummy_xf is not None else None,   # futr_exog
        dummy_xs,   # x_static
        True,
    )

    tx, step_fn = make_trainer(model, lr=lr, weight_decay=1e-5)
    opt_state = tx.init(params)

    losses = []
    dkey = random.PRNGKey(42)

    for s in range(1, steps + 1):
        dkey, sk = random.split(dkey)
        params, opt_state, loss = step_fn(params, opt_state, y_hist, x_f_all, x_static, sk)
        losses.append(float(loss))
        if verbose and (s % 100 == 0 or s == 1):
            print(f" step {s:4d} | nll {float(loss):.4f}")

    return model, params, losses


# -------------------------
# Inference
# -------------------------


def forecast_mc(
    params,
    model: DeepAR_EncDec,
    y_hist: jnp.ndarray,
    x_f_hist: jnp.ndarray = None,
    x_f_future: jnp.ndarray = None,
    x_static: jnp.ndarray = None,
    H: int = 24,
    N: int = 1000,
    seed: int = 2025,
) -> jnp.ndarray:
    """Monte Carlo probabilistic forecast.

    Encodes history, then samples N autoregressive trajectories of length H.

    Args:
        params: Trained model parameters.
        model: DeepAR_EncDec instance.
        y_hist: Historical values [T].
        x_f_future: Future exogenous features [H, F] or None.
        x_static: Static features [S] or None.
        H: Forecast horizon.
        N: Number of Monte Carlo sample paths.
        seed: Random seed.

    Returns:
        Sample paths [N, H].
    """
    model_cls = type(model)

    # Encode history 
    hT, cT = model.apply(
        params, y_hist, x_f_hist, x_static, method=model_cls.encode
    )

    if x_f_future is None:
        x_f_future = jnp.zeros((H, 0), dtype=jnp.float32)

    @jit
    def sample_one_path(key, y_last, h0, c0):
        def step_fn(carry, inputs):
            (y_prev, h, c), (x_f_step, k) = carry, inputs
            mu, sigma, h_new, c_new = model.apply(
                params,
                y_prev,
                x_f_step,
                x_static,
                h,
                c,
                True,
                method=model_cls.one_step,
            )
            eps = random.normal(k, (), dtype=jnp.float32)
            y_next = mu + sigma * eps
            return (y_next, h_new, c_new), y_next

        keys = random.split(key, H)
        _, samples = jax.lax.scan(step_fn, (y_last, h0, c0), (x_f_future, keys))
        return samples

    base_key = random.PRNGKey(seed)
    path_keys = random.split(base_key, N)
    paths = jax.vmap(lambda k: sample_one_path(k, y_hist[-1], hT, cT))(path_keys)
    return paths  # [N, H]


# -------------------------
# Quantile extraction
# -------------------------


def quantiles(paths: jnp.ndarray, qs=(0.1, 0.5, 0.9)):
    """Per-time quantiles from Monte Carlo paths.

    Args:
        paths: Sample paths [N, H].
        qs: Quantile levels.

    Returns:
        List of [H] arrays, one per quantile.
    """
    return [jnp.quantile(paths, q, axis=0) for q in qs]
