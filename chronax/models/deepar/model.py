"""DeepAR Encoder-Decoder model (FLAX/JAX/OPTAX implementation)."""

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import flax.linen as nn
import optax


def nll_gauss(y, mu, sigma):
    sigma = jnp.clip(sigma, 1e-6, 1e6)
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2


# ── Lag feature helpers ────────────────────────────────────────────────────────

_SHORT_LAGS = (1, 7)     # always-tracked lags (lag-1: autocorr; lag-7: weekly)
_BUF_SIZE   = 8          # need 8 history values to serve lag-1 and lag-7 during decode


def _build_lag_features(y: jnp.ndarray, h: int) -> jnp.ndarray:
    """[L, 4]: lag-1, lag-7, lag-h, lag-2h — zero-padded at start."""
    def lag(k):
        return jnp.concatenate([jnp.zeros(k, dtype=y.dtype), y])[:-k]
    return jnp.stack([lag(1), lag(7), lag(h), lag(2 * h)], axis=-1)


def _forecast_lags(y_hist: jnp.ndarray, enc_len: int, H: int):
    """
    Returns:
        xf_enc      [enc_len, 4]  — lag-1/7/H/2H features for the encoder
        xf_dec_safe [H, 2]        — lag-H/2H for the decoder (always from history)
        buf_init    [_BUF_SIZE]   — last _BUF_SIZE history values for decoder lag-1/7
    """
    max_k = max(2 * H, max(_SHORT_LAGS))
    T = y_hist.shape[0]
    pad = max(0, enc_len + max_k - T)
    y = jnp.concatenate([jnp.zeros(pad, dtype=y_hist.dtype), y_hist])
    n = y.shape[0]

    xf_enc = jnp.stack([
        y[n - enc_len - 1  : n - 1   ],   # lag-1
        y[n - enc_len - 7  : n - 7   ],   # lag-7
        y[n - enc_len - H  : n - H   ],   # lag-H
        y[n - enc_len - 2*H: n - 2*H ],   # lag-2H
    ], axis=-1)                            # [enc_len, 4]

    xf_dec_safe = jnp.stack([
        y[n - H - 1  : n - 1      ],      # lag-H at decode steps 0..H-1
        y[n - 2*H - 1: n - H - 1  ],      # lag-2H
    ], axis=-1)                            # [H, 2]

    # Buffer: last _BUF_SIZE history values (lag-1 = buf[-2], lag-7 = buf[0])
    buf_pad = max(0, _BUF_SIZE - T)
    buf_init = jnp.concatenate([
        jnp.zeros(buf_pad, dtype=y_hist.dtype), y_hist
    ])[-_BUF_SIZE:]                        # [_BUF_SIZE]

    return xf_enc, xf_dec_safe, buf_init


# ── Model ─────────────────────────────────────────────────────────────────────

class DeepAR_EncDec(nn.Module):
    hidden: int = 64
    dropout_rate: float = 0.1
    min_sigma: float = 0.02

    def setup(self):
        self.proj = nn.Dense(self.hidden)
        self.lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, out_axes=1,
        )(features=self.hidden)
        self.dropout  = nn.Dropout(rate=self.dropout_rate)
        self.head_mu  = nn.Dense(1)
        self.head_sig = nn.Dense(1)

    def encode(self, y_hist, futr_exog=None, x_static=None):
        T = y_hist.shape[0]
        parts = [y_hist[:, None]]
        if futr_exog is not None: parts.append(futr_exog)
        if x_static  is not None: parts.append(jnp.tile(x_static[None, :], (T, 1)))
        h0 = jnp.zeros((1, self.hidden), jnp.float32)
        c0 = jnp.zeros((1, self.hidden), jnp.float32)
        (hT, cT), _ = self.lstm((h0, c0), self.proj(jnp.concatenate(parts, -1)[None]))
        return hT, cT

    def one_step(self, y_prev, x_f_step=None, x_static=None,
                 h=None, c=None, deterministic=True):
        if h is None: h = jnp.zeros((1, self.hidden))
        if c is None: c = jnp.zeros((1, self.hidden))
        parts = [y_prev[None]]
        if x_f_step is not None: parts.append(x_f_step)
        if x_static  is not None: parts.append(x_static)
        x_vec = jnp.concatenate(parts, 0)[None, None, :]
        (h_new, c_new), hs = self.lstm((h, c), self.proj(x_vec))
        z     = self.dropout(hs[:, -1, :], deterministic=deterministic)
        mu    = self.head_mu(z)[..., 0]
        sigma = nn.softplus(self.head_sig(z)[..., 0]) + jnp.maximum(self.min_sigma, 1e-3)
        return mu[0], sigma[0], h_new, c_new

    def __call__(self, y_seq, futr_exog=None, x_static=None,
                 training=True, init_state=None):
        L  = y_seq.shape[0]
        xs = (jnp.zeros((L, 0), jnp.float32) if x_static is None
              else jnp.tile(x_static[None, :], (L, 1)))
        parts = [y_seq[:, None]]
        if futr_exog is not None: parts.append(futr_exog)
        parts.append(xs)
        z0 = self.dropout(self.proj(jnp.concatenate(parts, -1)[None]),
                          deterministic=not training)
        h0, c0 = init_state if init_state is not None else (
            jnp.zeros((1, self.hidden), jnp.float32),
            jnp.zeros((1, self.hidden), jnp.float32))
        (_, _), hs = self.lstm((h0, c0), z0)
        hs    = self.dropout(hs.squeeze(0), deterministic=not training)
        mu    = self.head_mu(hs)[..., 0]
        sigma = nn.softplus(self.head_sig(hs)[..., 0]) + jnp.maximum(self.min_sigma, 1e-3)
        return mu, sigma


# ── Legacy trainer ────────────────────────────────────────────────────────────

def make_trainer(model, lr=1e-3, weight_decay=1e-5):
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    def loss_fn(params, y_hist, futr_exog, x_static, key):
        mu, sg = model.apply(params, y_hist[:-1],
                             futr_exog[1:] if futr_exog is not None else None,
                             x_static, True, rngs={"dropout": key})
        return jnp.mean(nll_gauss(y_hist[1:], mu, sg))
    @jit
    def step(params, opt_state, y_hist, futr_exog, x_static, key):
        loss, grads = value_and_grad(loss_fn)(params, y_hist, futr_exog, x_static, key)
        upd, os2 = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), os2, loss
    return tx, step


# ── Module-level JIT cache ─────────────────────────────────────────────────────
# Each entry stores a compiled (vmap+scan) path-sampler for a model config.
# The scan carry includes a rolling lag buffer for lag-1 and lag-7 tracking.

_FORECAST_CACHE: dict = {}


def _get_forecast_fn(model: DeepAR_EncDec, H: int):
    key = (type(model).__name__, model.hidden, model.dropout_rate, model.min_sigma, H)
    if key not in _FORECAST_CACHE:
        mc = type(model)

        def _run(params, path_keys, y_last, h0, c0, buf_init, xf_dec_safe, x_static):
            # buf_init:    [_BUF_SIZE]  — last _BUF_SIZE history values
            # xf_dec_safe: [H, 2]       — precomputed lag-H and lag-2H
            # Decoder xf_step = [lag1, lag7, lagH, lag2H] ← dim 4 (matches training)
            def step_fn(carry, inp):
                (y_prev, h, c, buf), (xf_safe, k) = carry, inp
                lag1 = buf[-2]   # lag-1 relative to y_prev  (buf[-1] == y_prev)
                lag7 = buf[0]    # lag-7 (oldest element in 8-element buffer)
                xf   = jnp.concatenate([jnp.array([lag1, lag7]), xf_safe])  # [4]
                mu, sigma, h_new, c_new = model.apply(
                    params, y_prev, xf, x_static, h, c, True, method=mc.one_step)
                y_next  = mu + sigma * random.normal(k, (), jnp.float32)
                buf_new = jnp.concatenate([buf[1:], y_next[None]])  # shift & append
                return (y_next, h_new, c_new, buf_new), y_next

            def sample_one(key):
                _, s = jax.lax.scan(step_fn,
                                    (y_last, h0, c0, buf_init),
                                    (xf_dec_safe, random.split(key, H)))
                return s

            return jax.vmap(sample_one)(path_keys)

        _FORECAST_CACHE[key] = jax.jit(_run)
    return _FORECAST_CACHE[key]


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    y_hist: jnp.ndarray,
    x_f_all: jnp.ndarray = None,
    x_static: jnp.ndarray = None,
    hidden: int = 64,
    lr: float = 1e-3,
    steps: int = 800,
    dropout: float = 0.1,
    min_sigma: float = 0.02,
    h: int = 24,
    batch_size: int = 32,
    verbose: bool = True,
):
    """Train DeepAR with batched encode-decode and seasonal lag features.

    Lag features: lag-1 (autocorr), lag-7 (weekly), lag-h (horizon), lag-2h.
    During training, all lags are precomputed from teacher-forced windows.
    During inference, lag-1/7 are tracked via a rolling buffer in the scan carry.

    Returns (model, params, losses, scaler) where scaler = (mean, std).
    """
    y_mean = float(jnp.mean(y_hist))
    y_std  = float(jnp.std(y_hist)) + 1e-8
    y_norm = (y_hist - y_mean) / y_std

    L   = y_norm.shape[0]
    d_s = 0 if x_static is None else int(x_static.shape[0])

    # 4-feature lag matrix [L, 4]: lag-1, lag-7, lag-h, lag-2h
    xf_lags = _build_lag_features(y_norm, h)
    xf_all  = (jnp.concatenate([x_f_all, xf_lags], -1)
               if x_f_all is not None else xf_lags)
    d_f = int(xf_all.shape[1])   # 4 (or more if x_f_all provided)

    model     = DeepAR_EncDec(hidden=hidden, dropout_rate=dropout, min_sigma=min_sigma)
    model_cls = type(model)

    ctx_len = min(2 * h, max(1, L - h))
    dec_len = h
    window  = ctx_len + dec_len
    n_wins  = max(1, L - window + 1)
    eff_b   = min(batch_size, n_wins)

    key      = random.PRNGKey(1)
    dummy_xf = jnp.zeros((ctx_len, d_f), jnp.float32)
    dummy_xs = jnp.zeros(d_s, jnp.float32) if d_s > 0 else None

    params = model.init({"params": key, "dropout": key},
                        jnp.ones(ctx_len, jnp.float32), dummy_xf, dummy_xs, True)

    warmup   = max(10, steps // 10)
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, lr * 0.01)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(learning_rate=schedule, weight_decay=1e-5))

    def _wloss(params, y_win, xf_win, key):
        h0, c0 = model.apply(params, y_win[:ctx_len], xf_win[:ctx_len],
                              x_static, method=model_cls.encode)
        y_in  = y_win[ctx_len - 1: ctx_len - 1 + dec_len]
        y_tgt = y_win[ctx_len: ctx_len + dec_len]
        mu, sg = model.apply(params, y_in, xf_win[ctx_len: ctx_len + dec_len],
                             x_static, True, init_state=(h0, c0), rngs={"dropout": key})
        return jnp.mean(nll_gauss(y_tgt, mu, sg))

    @jit
    def _batch_step(params, opt_state, key):
        wk, sk = random.split(key)
        starts  = random.randint(wk, (eff_b,), 0, n_wins)
        y_wins  = jax.vmap(lambda s: jax.lax.dynamic_slice(y_norm, (s,),      (window,)))(starts)
        xf_wins = jax.vmap(lambda s: jax.lax.dynamic_slice(xf_all, (s, 0), (window, d_f)))(starts)
        bkeys   = random.split(sk, eff_b)
        def bl(p):
            return jnp.mean(jax.vmap(lambda yw, xfw, k: _wloss(p, yw, xfw, k))
                            (y_wins, xf_wins, bkeys))
        loss, grads = value_and_grad(bl)(params)
        upd, new_opt = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), new_opt, loss

    opt_state = tx.init(params)
    losses    = []
    dkey      = random.PRNGKey(42)
    for s in range(1, steps + 1):
        dkey, sk = random.split(dkey)
        params, opt_state, loss = _batch_step(params, opt_state, sk)
        losses.append(float(loss))
        if verbose and (s % 100 == 0 or s == 1):
            print(f" step {s:4d} | nll {float(loss):.4f}")

    return model, params, losses, (y_mean, y_std)


# ── Inference ─────────────────────────────────────────────────────────────────

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
    scaler=None,
    input_size: int = None,
) -> jnp.ndarray:
    """Monte Carlo forecast. Returns sample paths [N, H].

    Lag-1/7 are tracked via a rolling buffer in the scan carry so predicted
    values are correctly used as lags in later decode steps.
    """
    if scaler is not None:
        y_mean, y_std = scaler
        y_hist = (y_hist - y_mean) / y_std

    enc_len = input_size if input_size is not None else y_hist.shape[0]
    y_enc   = y_hist[-enc_len:]

    # Compute lag features; buf_init carries last _BUF_SIZE normalised values
    xf_enc, xf_dec_safe, buf_init = _forecast_lags(y_hist, enc_len, H)

    xf_enc_hist = (x_f_hist[-enc_len:] if (x_f_hist is not None and input_size is not None)
                   else x_f_hist)
    xf_enc_final = (jnp.concatenate([xf_enc_hist, xf_enc], -1)
                    if xf_enc_hist is not None else xf_enc)        # [enc_len, 4]

    # x_f_future (user) merged with safe lags only (lag-1/7 come from buffer at runtime)
    xf_fut_safe = (jnp.concatenate([x_f_future, xf_dec_safe], -1)
                   if x_f_future is not None else xf_dec_safe)     # [H, 2]

    model_cls = type(model)
    # Encode all but the last step; decode starts from y_enc[-1] (no double-feed).
    hT, cT = model.apply(
        params, y_enc[:-1], xf_enc_final[:-1], x_static, method=model_cls.encode
    )

    path_keys = random.split(random.PRNGKey(seed), N)
    sample_fn = _get_forecast_fn(model, H)
    paths     = sample_fn(params, path_keys, y_enc[-1], hT, cT,
                          buf_init, xf_fut_safe, x_static)

    if scaler is not None:
        paths = paths * y_std + y_mean
    return paths


def forecast_point(
    params,
    model: DeepAR_EncDec,
    y_hist: jnp.ndarray,
    x_f_hist: jnp.ndarray = None,
    x_f_future: jnp.ndarray = None,
    x_static: jnp.ndarray = None,
    H: int = 24,
    scaler=None,
    input_size: int = None,
) -> jnp.ndarray:
    """Deterministic μ-feedback forecast (MAE-aligned point estimate) [H]."""
    if scaler is not None:
        y_mean, y_std = scaler
        y_hist = (y_hist - y_mean) / y_std

    enc_len = input_size if input_size is not None else y_hist.shape[0]
    y_enc = y_hist[-enc_len:]
    xf_enc, xf_dec_safe, buf_init = _forecast_lags(y_hist, enc_len, H)

    xf_enc_hist = (x_f_hist[-enc_len:] if (x_f_hist is not None and input_size is not None)
                   else x_f_hist)
    xf_enc_final = (jnp.concatenate([xf_enc_hist, xf_enc], -1)
                    if xf_enc_hist is not None else xf_enc)
    xf_fut_safe = (jnp.concatenate([x_f_future, xf_dec_safe], -1)
                   if x_f_future is not None else xf_dec_safe)

    model_cls = type(model)
    hT, cT = model.apply(
        params, y_enc[:-1], xf_enc_final[:-1], x_static, method=model_cls.encode
    )

    def step_fn(carry, xf_safe):
        y_prev, h, c, buf = carry
        lag1 = buf[-2]
        lag7 = buf[0]
        xf = jnp.concatenate([jnp.array([lag1, lag7]), xf_safe])
        mu, _sigma, h_new, c_new = model.apply(
            params, y_prev, xf, x_static, h, c, True, method=model_cls.one_step
        )
        buf_new = jnp.concatenate([buf[1:], mu[None]])
        return (mu, h_new, c_new, buf_new), mu

    (_y, _h, _c, _buf), path = jax.lax.scan(
        step_fn, (y_enc[-1], hT, cT, buf_init), xf_fut_safe
    )
    if scaler is not None:
        path = path * y_std + y_mean
    return path


def quantiles(paths: jnp.ndarray, qs=(0.1, 0.5, 0.9)):
    return [jnp.quantile(paths, q, axis=0) for q in qs]
