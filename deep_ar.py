# DeepAR with horizon-wise calibrated uncertainty and robust sigma floor
# ---------------------------------------------------------------------
# 1) Sigma floor set from *noise* (volatility), not level.
# 2) Mild horizon-aware σ inflation: sqrt(1 + γ·h).
# 3) Per-horizon in-sample calibration via free-roll backtesting:
#    compute f[h] so the 80% interval calibrates at each horizon h.
# 4) AdamW + dropout; trainer.step(..., key=None) is legacy-compatible.
# 5) Test-harness friendly returns (has_nan_loss / has_nan_forecast).

import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
import flax.linen as nn
import optax


# -------------------------
# Synthetic series generators (pure JAX / Python)
# -------------------------

def make_series(T=200, H=24, seed=0):
    """Original series with seasonality and trend"""
    key = random.PRNGKey(seed)
    eps = 0.3 * random.normal(key, (T + H,))
    y = jnp.zeros(T + H, dtype=jnp.float32)
    def body_fun(i, y_):
        val = (
            0.6 * y_[i - 1]
            + 0.9 * jnp.sin(2 * jnp.pi * i / 12.0)
            + 0.35 * jnp.cos(2 * jnp.pi * i / 12.0)
            + 0.02 * i
            + eps[i]
        )
        return y_.at[i].set(val)
    y = jax.lax.fori_loop(1, T + H, body_fun, y)
    return y[:T].astype(jnp.float32), y[T:].astype(jnp.float32)


def make_linear_trend(T=200, H=24, seed=0):
    """Simple linear trend with noise"""
    key = random.PRNGKey(seed)
    t = jnp.arange(T + H)
    eps = 0.5 * random.normal(key, (T + H,))
    y = 5.0 + 0.1 * t + eps
    return y[:T].astype(jnp.float32), y[T:].astype(jnp.float32)


def make_seasonal_only(T=200, H=24, seed=0):
    """Pure seasonality, no trend"""
    key = random.PRNGKey(seed)
    t = jnp.arange(T + H)
    eps = 0.2 * random.normal(key, (T + H,))
    y = 10.0 + 3.0 * jnp.sin(2 * jnp.pi * t / 24.0) + eps
    return y[:T].astype(jnp.float32), y[T:].astype(jnp.float32)


def make_volatile_series(T=200, H=24, seed=0):
    """High volatility series"""
    key = random.PRNGKey(seed)
    eps = 1.5 * random.normal(key, (T + H,))
    y = jnp.zeros(T + H, dtype=jnp.float32)
    def body_fun(i, y_):
        val = 0.3 * y_[i - 1] + 2.0 * jnp.sin(2 * jnp.pi * i / 15.0) + eps[i]
        return y_.at[i].set(val)
    y = jax.lax.fori_loop(1, T + H, body_fun, y)
    return y[:T].astype(jnp.float32), y[T:].astype(jnp.float32)


def make_step_change(T=200, H=24, seed=0):
    """Series with a step change in the middle"""
    key = random.PRNGKey(seed)
    eps = 0.3 * random.normal(key, (T + H,))
    y = jnp.zeros(T + H, dtype=jnp.float32)
    def body_fun(i, y_):
        base = jnp.where(i < 150, 10.0, 20.0)
        val = 0.5 * y_[i - 1] + base + eps[i]
        return y_.at[i].set(val)
    y = jax.lax.fori_loop(1, T + H, body_fun, y)
    return y[:T].astype(jnp.float32), y[T:].astype(jnp.float32)


# -------------------------
# Model
# -------------------------
class MiniDeepAR(nn.Module):
    """
    Minimal DeepAR-style model with calibrated uncertainty.

    Parameters
    ----------
    hidden : int
        LSTM hidden size.
    dropout_rate : float
        Dropout rate (training only).
    min_sigma_scale : float
        Sigma floor *in scaled space*; typically noise_scale / level_scale,
        but we also clip to a small absolute minimum to avoid collapse.
    horizon_coeff : float
        Coefficient γ for sqrt(1 + γ·h) horizon inflation.
    """
    hidden: int = 64
    dropout_rate: float = 0.1
    min_sigma_scale: float = 0.05
    horizon_coeff: float = 0.06  # a bit stronger to help low coverage

    def setup(self):
        self.lstm = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, out_axes=1,
        )(features=self.hidden)

        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Heads
        self.head_mu = nn.Dense(
            1,
            kernel_init=nn.initializers.variance_scaling(
                scale=0.1, mode='fan_in', distribution='truncated_normal'
            )
        )
        self.head_sigma = nn.Dense(
            1,
            kernel_init=nn.initializers.constant(0.0),
            bias_init=nn.initializers.constant(0.0)  # neutral start
        )

    @nn.compact
    def __call__(self, y_in: jnp.ndarray, training: bool = False):
        """
        y_in: (L,) teacher-forced input
        returns: mu, sigma of shape (L,)
        """
        x_seq = y_in[None, :, None]  # (1, L, 1)
        B, L, _ = x_seq.shape

        h0 = jnp.zeros((B, self.hidden))
        c0 = jnp.zeros((B, self.hidden))
        (hT, cT), hs = self.lstm((h0, c0), x_seq)  # hs: (B, L, hidden)
        hs = hs.squeeze(0)  # (L, hidden)

        hs = self.dropout(hs, deterministic=not training)

        mu = self.head_mu(hs)[..., 0]
        sraw = self.head_sigma(hs)[..., 0]
        sigma = nn.softplus(sraw) + jnp.maximum(self.min_sigma_scale, 1e-3)
        return mu, sigma

    @nn.compact
    def condition(self, y_hist: jnp.ndarray):
        """Return final LSTM state after consuming history."""
        x_seq = y_hist[None, :, None]
        h0 = jnp.zeros((1, self.hidden))
        c0 = jnp.zeros((1, self.hidden))
        (hT, cT), _ = self.lstm((h0, c0), x_seq)
        return hT, cT

    @nn.compact
    def one_step(self, y_prev_scalar: jnp.ndarray, h: jnp.ndarray, c: jnp.ndarray,
                 step_ahead: int = 1):
        """
        Autoregressive single step with mild horizon-aware σ growth.
        """
        x1 = y_prev_scalar[None, None, None]
        (h_new, c_new), hs = self.lstm((h, c), x1)
        h_out = hs[:, -1, :]

        mu = self.head_mu(h_out)[..., 0]
        sraw = self.head_sigma(h_out)[..., 0]
        sigma_base = nn.softplus(sraw) + jnp.maximum(self.min_sigma_scale, 1e-3)

        horizon_scale = jnp.sqrt(1.0 + self.horizon_coeff * step_ahead)
        sigma = sigma_base * horizon_scale

        return mu[0], sigma[0], h_new, c_new


ImprovedDeepAR = MiniDeepAR  # compatibility alias


# -------------------------
# NLL and trainer
# -------------------------
def nll_gauss(y, mu, sigma):
    sigma = jnp.clip(sigma, 1e-6, 1e6)
    return 0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2


def make_trainer(model, lr=1e-3, weight_decay=1e-5):
    """
    Create AdamW optimizer and training step function.
    
    Parameters
    ----------
    model : MiniDeepAR
        The model instance
    lr : float
        Learning rate
    weight_decay : float
        L2 regularization coefficient
        
    Returns
    -------
    tx : optax.GradientTransformation
        The optimizer
    step : callable
        Training step function with signature:
        step(params, opt_state, y_hist, key=None) -> (params, opt_state, loss)
    """
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(params, y_hist, key):
        """
        Compute negative log-likelihood loss with teacher forcing.
        
        The model predicts y[t] given y[t-1], so we compute NLL
        for predictions mu[:-1] against targets y[1:].
        """
        mu, sigma = model.apply(params, y_hist, training=True, rngs={'dropout': key})
        
        # One-step-ahead NLL with teacher forcing
        nll = jnp.mean(nll_gauss(y_hist[1:], mu[:-1], sigma[:-1]))
        
        # No additional sigma penalty - let min_sigma_scale handle the floor
        # (Removed redundant penalty that conflicted with min_sigma_scale)
        return nll

    @jit
    def step(params, opt_state, y_hist, key=None):
        """
        Perform one training step.
        
        Parameters
        ----------
        params : PyTree
            Current model parameters
        opt_state : optax.OptState
            Current optimizer state
        y_hist : jnp.ndarray
            Training sequence (scaled)
        key : jax.random.PRNGKey, optional
            Random key for dropout. If None, uses fixed seed.
            
        Returns
        -------
        params : PyTree
            Updated parameters
        opt_state : optax.OptState
            Updated optimizer state
        loss : float
            Loss value for this step
        """
        if key is None:
            key = random.PRNGKey(0)
        
        loss, grads = value_and_grad(loss_fn)(params, y_hist, key)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss

    return tx, step


# -------------------------
# Horizon-wise calibration via free-roll backtest
# -------------------------
def _free_roll_metrics(params, model, y_hist, Hc=24, K=8):
    """
    Free-roll forecasts to gather |resid|/σ stats per horizon 1..Hc.
    Returns list[r_h], each r_h is jnp.array of ratios for that horizon.
    """
    L = int(y_hist.shape[0])
    if L < Hc + 5:
        return [jnp.array([], dtype=jnp.float32) for _ in range(Hc)]

    last_anchor_end = L - Hc - 1
    first_anchor = max(1, last_anchor_end - K + 1)
    anchors = list(range(first_anchor, last_anchor_end + 1))
    r_by_h = [[] for _ in range(Hc)]

    model_class = type(model)

    for a in anchors:
        hist = y_hist[:a+1]
        hT, cT = model.apply(params, hist, method=model_class.condition)
        y_prev = hist[-1]
        h, c = hT, cT

        for hidx in range(1, Hc + 1):
            mu, sigma, h, c = model.apply(params, y_prev, h, c, hidx, method=model_class.one_step)
            t = a + hidx
            if t < L:
                resid = jnp.abs(y_hist[t] - mu)
                denom = jnp.maximum(sigma, 1e-6)
                r_by_h[hidx - 1].append(float(resid / denom))
            # deterministic free-roll step
            y_prev = mu

    return [jnp.array(rs, dtype=jnp.float32) if rs else jnp.array([], dtype=jnp.float32)
            for rs in r_by_h]


def _compute_sigma_calibration_per_h(params, model, y_hist, H):
    """
    Per-horizon sigma calibration multipliers via free-roll backtesting.
    
    Computes f[h] such that the empirical 90th percentile of |resid|/σ 
    matches the theoretical value (1.28) for an 80% prediction interval.
    
    For horizons beyond H_eff (default 24), we extrapolate using the last
    calibrated value to maintain consistency.
    
    Parameters
    ----------
    params : PyTree
        Model parameters
    model : MiniDeepAR
        The model instance
    y_hist : jnp.ndarray
        Historical time series (scaled)
    H : int
        Total forecast horizon
        
    Returns
    -------
    f : jnp.ndarray of shape (H,)
        Per-horizon calibration multipliers
    """
    Hc = int(H)
    H_eff = min(Hc, 24)  # Calibrate up to 24 horizons
    
    # Gather free-roll statistics
    K = min(12, max(5, int(y_hist.shape[0]) // 20))  # Adaptive number of anchors
    r_by_h = _free_roll_metrics(params, model, y_hist, Hc=H_eff, K=K)
    
    # Target: 90th percentile should equal 1.28155 for 80% coverage
    z90 = 1.281551565545

    # Initialize all multipliers to 1.0 (neutral)
    f = jnp.ones((Hc,), dtype=jnp.float32)
    f_list = []
    
    for h in range(H_eff):
        rs = r_by_h[h]
        if rs.size >= 5:  # Need at least 5 samples for reliable quantile
            r90 = jnp.quantile(rs, 0.90)
            f_h = r90 / z90
            # Conservative clipping: allow slight deflation but prefer wider intervals
            f_h = jnp.clip(f_h, 0.90, 2.0)
        else:
            # Neutral fallback when insufficient data
            f_h = jnp.array(1.0, dtype=jnp.float32)
        f_list.append(f_h)

    if H_eff > 0:
        f = f.at[:H_eff].set(jnp.stack(f_list))
        
        # Extrapolate beyond H_eff using the last calibrated value
        if Hc > H_eff:
            f = f.at[H_eff:].set(f_list[-1])

    return f


# -------------------------
# Forecasting
# -------------------------
def forecast_mc(params, model, y_hist, H=24, N=1000, seed=2025, calibrate=True):
    """
    Monte Carlo forecasting with optional per-horizon σ calibration.
    """
    if calibrate:
        fph = _compute_sigma_calibration_per_h(params, model, y_hist, H)
    else:
        fph = jnp.ones((H,), dtype=jnp.float32)

    model_class = type(model)
    hT, cT = model.apply(params, y_hist, method=model_class.condition)

    len_fph_minus1 = jnp.int32(fph.shape[0] - 1)

    @jit
    def forecast_one_path(key, h0, c0, y_last):
        def step_fn(carry, step_key):
            y_prev, h, c, step_idx = carry
            mu, sigma, h_new, c_new = model.apply(
                params, y_prev, h, c, step_idx, method=model_class.one_step
            )
            idx = jnp.minimum(step_idx - 1, len_fph_minus1)
            sigma_adj = sigma * fph[idx]
            eps = random.normal(step_key, ())
            y_next = mu + sigma_adj * eps
            return (y_next, h_new, c_new, step_idx + 1), y_next

        keys = random.split(key, H)
        _, samples = jax.lax.scan(
            step_fn,
            (y_last, h0, c0, jnp.array(1, dtype=jnp.int32)),
            keys
        )
        return samples  # (H,)

    key = random.PRNGKey(seed)
    y_last = y_hist[-1]
    path_keys = random.split(key, N)
    paths = jax.vmap(lambda k: forecast_one_path(k, hT, cT, y_last))(path_keys)
    return paths  # (N, H) jnp.array


# -------------------------
# Training
# -------------------------
def train_model(y_scaled, hidden_size=64, lr=1e-3, steps=800,
                dropout_rate=0.1, min_sigma_scale=0.05, horizon_coeff=0.04,
                verbose=True):
    """
    Train the model. min_sigma_scale is in *scaled* units.
    """
    model = ImprovedDeepAR(
        hidden=hidden_size,
        dropout_rate=dropout_rate,
        min_sigma_scale=min_sigma_scale,
        horizon_coeff=horizon_coeff,
    )

    key = random.PRNGKey(1)
    init_key, dropout_key = random.split(key)
    params = model.init({'params': init_key, 'dropout': dropout_key},
                        jnp.array(y_scaled), training=True)

    tx, step_fn = make_trainer(model, lr=lr, weight_decay=1e-5)
    opt_state = tx.init(params)

    losses = []
    step_key = random.PRNGKey(42)
    for s in range(1, steps + 1):
        step_key, sk = random.split(step_key)
        params, opt_state, l = step_fn(params, opt_state, jnp.array(y_scaled), sk)
        if verbose and (s % 100 == 0 or s == 1):
            print(f"  step {s:4d} | loss {float(l):.4f}")
        losses.append(float(l))

    return model, params, losses


# -------------------------
# Convenience wrapper for testing/running (NumPy-free)
# -------------------------
def run_test(test_name, series_fn, T=200, H=24, N=1000, steps=800,
             hidden_size=64, dropout_rate=0.1,
             horizon_coeff=0.06, seed=0, verbose=True):
    """
    Full pipeline (NumPy-free):
      - generate data
      - compute level & noise scales
      - train (in scaled space)
      - forecast with per-horizon σ calibration
      - return quantiles and metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")

    # Data
    y_hist, y_true_future = series_fn(T=T, H=H, seed=seed)

    # Level scale for normalization
    level_scale = jnp.maximum(1e-3, jnp.mean(jnp.abs(y_hist)))

    # Robust noise scale for sigma floor (level-invariant)
    dy = jnp.diff(y_hist, prepend=y_hist[0])
    mad = jnp.median(jnp.abs(dy - jnp.median(dy)))
    noise_scale = jnp.maximum(1e-6, 1.4826 * mad)

    # Prepare scaled training series
    y_scaled = (y_hist / level_scale).astype(jnp.float32)

    # Sigma floor in scaled space = noise_scale / level_scale, also clip to ≥ 0.02
    min_sigma_scale = jnp.maximum(noise_scale / level_scale, 0.02)

    if verbose:
        print(f"Scaling: level_scale={float(level_scale):.4g}, noise_scale={float(noise_scale):.4g}, "
              f"min_sigma_scale(scaled)={float(min_sigma_scale):.4g}")

    # Train
    if verbose:
        print("Training...")
    model, params, losses = train_model(
        y_scaled,
        hidden_size=hidden_size,
        steps=steps,
        dropout_rate=dropout_rate,
        min_sigma_scale=float(min_sigma_scale),
        horizon_coeff=horizon_coeff,
        verbose=verbose
    )

    # Forecast (calibrated), then unscale
    if verbose:
        print(f"Forecasting {N} paths...")
    paths_scaled = forecast_mc(params, model, jnp.array(y_scaled), H=H, N=N, seed=2025, calibrate=True)
    paths = paths_scaled * level_scale  # (N, H)

    # Quantiles (NumPy-free)
    q10 = jnp.quantile(paths, 0.10, axis=0)
    q50 = jnp.quantile(paths, 0.50, axis=0)
    q90 = jnp.quantile(paths, 0.90, axis=0)

    # Metrics
    mae = jnp.mean(jnp.abs(q50 - y_true_future))
    coverage = jnp.mean((y_true_future >= q10) & (y_true_future <= q90))

    if verbose:
        print(f"✓ MAE (median forecast): {float(mae):.3f}")
        print(f"✓ 80% Coverage: {float(coverage)*100:.1f}%")

    # NaN flags
    has_nan_loss = bool(jnp.any(jnp.isnan(jnp.array(losses))))
    has_nan_forecast = bool(jnp.any(jnp.isnan(paths)))

    return {
        'y_hist': y_hist,
        'y_true_future': y_true_future,
        'q10': q10, 'q50': q50, 'q90': q90,
        'losses': losses,
        'mae': float(mae),
        'coverage': float(coverage),
        'has_nan_loss': has_nan_loss,
        'has_nan_forecast': has_nan_forecast
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Improved DeepAR Demo (NumPy-free; noise-based σ floor + per-horizon calibration)")
    print("=" * 60)

    test_cases = {
        "Original (Seasonal+Trend+AR)": make_series,
        "Linear Trend": make_linear_trend,
        "Pure Seasonal": make_seasonal_only,
    }

    for test_name, series_fn in test_cases.items():
        _ = run_test(
            test_name=test_name,
            series_fn=series_fn,
            T=200,
            H=24,
            N=1000,
            steps=600,
            hidden_size=64,
            dropout_rate=0.1,
            horizon_coeff=0.04,
            seed=42,
            verbose=True
        )

    print("\n" + "=" * 60)
    print("✓ All tests complete!")
    print("=" * 60)