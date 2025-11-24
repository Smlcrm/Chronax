# Minimal Encoder–Decoder DeepAR (NumPy-free, JAX/Flax)
# -----------------------------------------------------
# Comments added throughout for clarity.

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import flax.linen as nn
import optax

# -------------------------
# Utilities
# -------------------------

def _safe_cat(a, b):
    """
    Concatenate a and b handling None.
    Returns 0-dummy vector if both None.
    """
    if a is None and b is None:
        return jnp.zeros((0,), dtype=jnp.float32)
    if a is None:
        return b
    if b is None:
        return a
    return jnp.concatenate([a, b], axis=-1)


def nll_gauss(y, mu, sigma):
    """
    Gaussian negative log-likelihood.
    sigma clipped for numerical safety.
    """
    sigma = jnp.clip(sigma, 1e-6, 1e6)
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2


# -------------------------
# Model
# -------------------------

class DeepAR_EncDec(nn.Module):
    """
    Encoder–Decoder DeepAR model implemented in Flax.

    - Encoder LSTM consumes history y_{1:T}
      (plus optional static features)
      to produce hidden state (h_T, c_T).

    - Decoder one_step() takes [y_prev, x_f, x_static]
      and produces Gaussian parameters (mu, sigma).

    - training_roll() unrolls decoder with teacher forcing.
    """
    hidden: int = 64
    dropout_rate: float = 0.1
    min_sigma: float = 0.02

    def setup(self):
        # Separate projections for encoder and decoder inputs
        self.enc_proj = nn.Dense(self.hidden)
        self.dec_proj = nn.Dense(self.hidden)

        # LSTM wrapped inside nn.scan to run over time dimension
        self.lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(features=self.hidden)

        # Dropout for regularization
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Output heads for Gaussian parameters
        self.head_mu = nn.Dense(1)
        self.head_sigma = nn.Dense(1)

    # ---------- Encoder ----------
    @nn.compact
    def encode(self, y_hist: jnp.ndarray, x_static: jnp.ndarray = None):
        """
        Encode history y_hist (T,) + static to get (h_T, c_T).
        """
        T = y_hist.shape[0]

        # Repeat static features across time if provided
        if x_static is None:
            xs_rep = jnp.zeros((T, 0), dtype=jnp.float32)
        else:
            xs_rep = jnp.tile(x_static[None, :], (T, 1))

        # Encoder input = [y_t, x_static]
        x_enc = jnp.concatenate([y_hist[:, None], xs_rep], axis=-1)
        x_enc = x_enc[None, :, :]  # (1, T, D)

        # Initial hidden/cell states
        B = 1
        h0 = jnp.zeros((B, self.hidden))
        c0 = jnp.zeros((B, self.hidden))

        # Project encoder input
        x_proj = self.enc_proj(x_enc)

        # Run through LSTM
        (hT, cT), _ = self.lstm((h0, c0), x_proj)

        return hT, cT

    # ---------- Decoder one-step ----------
    @nn.compact
    def one_step(self,
                 y_prev_scalar: jnp.ndarray,
                 x_f_step: jnp.ndarray = None,
                 x_static: jnp.ndarray = None,
                 h: jnp.ndarray = None,
                 c: jnp.ndarray = None,
                 deterministic: bool = True):
        """
        One decoding step:
        Input = [y_prev, x_f_step, x_static]
        Output = (mu, sigma, h_new, c_new)
        """

        # If hidden states not provided, start from zero
        if h is None:
            h = jnp.zeros((1, self.hidden))
        if c is None:
            c = jnp.zeros((1, self.hidden))

        # Build concatenated input vector
        parts = [y_prev_scalar[None]]
        if x_f_step is not None:
            parts.append(x_f_step)
        if x_static is not None:
            parts.append(x_static)

        # Shape (1,1,D)
        x_vec = jnp.concatenate(parts, axis=0)[None, None, :]

        # Linear projection
        x_proj = self.dec_proj(x_vec)

        # LSTM step
        (h_new, c_new), hs = self.lstm((h, c), x_proj)

        # Extract hidden vector
        z = hs[:, -1, :]
        z = self.dropout(z, deterministic=deterministic)

        # Output Gaussian parameters
        mu = self.head_mu(z)[..., 0]
        sraw = self.head_sigma(z)[..., 0]
        sigma = nn.softplus(sraw) + jnp.maximum(self.min_sigma, 1e-3)

        return mu[0], sigma[0], h_new, c_new

    # ---------- Training roll with teacher forcing ----------
    @nn.compact
    def training_roll(self,
                      y_seq: jnp.ndarray,
                      x_f_all: jnp.ndarray = None,
                      x_static: jnp.ndarray = None,
                      training: bool = True):
        """
        Compute mu_t, sigma_t for t = 0..L-2 predicting y_{t+1}.

        Teacher forcing:
            Input at step t = [y_t, x_f_{t+1}, x_static]

        Encoder processes only y_seq[:-1] (history) to avoid leakage.
        """
        L = y_seq.shape[0]

        # Repeat static inputs over L-1 decoding steps
        if x_static is None:
            xs_rep = jnp.zeros((L - 1, 0), dtype=jnp.float32)
        else:
            xs_rep = jnp.tile(x_static[None, :], (L - 1, 1))

        # y_t (teacher forcing inputs)
        y_t = y_seq[:-1][:, None]

        # x_f_{t+1}
        if x_f_all is None:
            xf_tp1 = jnp.zeros((L - 1, 0), dtype=jnp.float32)
        else:
            xf_tp1 = x_f_all[1:]

        # Build decoder input matrix (L-1, D)
        dec_inputs = jnp.concatenate([y_t, xf_tp1, xs_rep], axis=-1)
        dec_inputs = dec_inputs[None, :, :]

        # Encode only the historical part
        hT, cT = self.encode(y_seq[:-1], x_static)

        # Projection + dropout
        x_proj = self.dec_proj(dec_inputs)
        z0 = self.dropout(x_proj, deterministic=not training)

        # LSTM unroll
        (h_final, c_final), hs = self.lstm((hT, cT), z0)
        hs = hs.squeeze(0)
        hs = self.dropout(hs, deterministic=not training)

        # Output Gaussian parameters per step
        mu = self.head_mu(hs)[..., 0]
        sraw = self.head_sigma(hs)[..., 0]
        sigma = nn.softplus(sraw) + jnp.maximum(self.min_sigma, 1e-3)

        return mu, sigma


# -------------------------
# Training wrapper
# -------------------------

def make_trainer(model, lr=1e-3, weight_decay=1e-5):
    """
    Build Optax optimizer + loss function.
    """
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(params, y_hist, x_f_all, x_static, key):
        # Forward pass through training_roll()
        mu, sigma = model.apply(
            params,
            y_hist,
            x_f_all,
            x_static,
            True,
            method=type(model).training_roll,
            rngs={'dropout': key}
        )
        # Targets = y[1:]
        y_target = y_hist[1:]
        nll = jnp.mean(nll_gauss(y_target, mu, sigma))
        return nll

    @jit
    def step(params, opt_state, y_hist, x_f_all, x_static, key=None):
        # Single optimizer step
        if key is None:
            key = random.PRNGKey(0)

        loss, grads = value_and_grad(loss_fn)(params, y_hist, x_f_all, x_static, key)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return tx, step


def train_model(y_hist,
                x_f_all=None,
                x_static=None,
                hidden=64,
                lr=1e-3,
                steps=800,
                dropout=0.1,
                min_sigma=0.02,
                verbose=True):
    """
    High-level training loop.
    """

    # Create model instance
    model = DeepAR_EncDec(hidden=hidden, dropout_rate=dropout, min_sigma=min_sigma)

    # Determine dimensions for initialization
    L = y_hist.shape[0]
    d_f = 0 if x_f_all is None else int(x_f_all.shape[1])
    d_s = 0 if x_static is None else int(x_static.shape[0])

    # Initialize parameters with dummy call
    key = random.PRNGKey(1)
    params = model.init(
        {'params': key, 'dropout': key},
        jnp.array(y_hist),
        None if x_f_all is None else jnp.zeros((L, d_f), dtype=jnp.float32),
        None if x_static is None else jnp.zeros((d_s,), dtype=jnp.float32),
        True,
        method=DeepAR_EncDec.training_roll
    )

    # Build optimizer
    tx, step_fn = make_trainer(model, lr=lr, weight_decay=1e-5)
    opt_state = tx.init(params)

    losses = []
    dkey = random.PRNGKey(42)

    # Training loop
    for s in range(1, steps + 1):
        dkey, sk = random.split(dkey)
        params, opt_state, loss = step_fn(params, opt_state, y_hist, x_f_all, x_static, sk)
        losses.append(float(loss))

        if verbose and (s % 100 == 0 or s == 1):
            print(f" step {s:4d} | nll {float(loss):.4f}")

    return model, params, losses


# -------------------------
# Forecasting
# -------------------------

def forecast_mc(params,
                model,
                y_hist,
                x_f_future=None,
                x_static=None,
                H=24,
                N=1000,
                seed=2025):
    """
    Monte Carlo forecast:

    1) Encode history -> initial (h_T, c_T)
    2) For each horizon step h:
       input = [y_{t+h-1}, x_f_future[h], x_static]
       sample y_{t+h} from N(mu, sigma)
    """

    model_cls = type(model)

    # Encode history
    hT, cT = model.apply(params, y_hist, x_static, method=model_cls.encode)

    # Ensure covariates exist
    if x_f_future is None:
        x_f_future = jnp.zeros((H, 0), dtype=jnp.float32)

    @jit
    def sample_one_path(key, y_last, h0, c0):
        """
        Draw one full forecast trajectory of length H.
        """
        def step_fn(carry, inputs):
            (y_prev, h, c), (x_f_step, k) = carry, inputs

            # Deterministic=True (no dropout)
            mu, sigma, h_new, c_new = model.apply(
                params,
                y_prev,
                x_f_step,
                x_static,
                h,
                c,
                True,
                method=model_cls.one_step
            )

            # Sample from Gaussian
            eps = random.normal(k, ())
            y_next = mu + sigma * eps
            return (y_next, h_new, c_new), y_next

        # Provide separate random keys per step
        keys = random.split(key, H)
        inputs = (x_f_future, keys)

        y0 = y_hist[-1]

        # Scan through decoder
        (_, _, _), samples = jax.lax.scan(
            step_fn,
            (y0, h0, c0),
            inputs
        )
        return samples

    # Sample N Monte Carlo trajectories
    base_key = random.PRNGKey(seed)
    path_keys = random.split(base_key, N)

    paths = jax.vmap(lambda k: sample_one_path(k, y_hist[-1], hT, cT))(path_keys)
    return paths


# -------------------------
# Quantiles
# -------------------------

def quantiles(paths, qs=(0.1, 0.5, 0.9)):
    """
    Compute per-time quantiles of MC forecast paths.
    """
    q_list = [jnp.quantile(paths, q, axis=0) for q in qs]
    return q_list
