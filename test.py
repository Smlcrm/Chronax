# run_ets_optimize_checks.py
import sys
import math
import numpy as np
import jax.numpy as jnp

import ets_backend as ets  # your module (the JAX port)

# -------------------------------------------------------------------
# Wrappers that call the SAME objective as the optimizer uses
# -------------------------------------------------------------------

def jax_objective_from_params(p, y, n_state,
                              error, trend, season,
                              opt_crit, n_mse, m,
                              opt_alpha, opt_beta, opt_gamma, opt_phi,
                              alpha, beta, gamma, phi) -> float:
    """Call the exact objective used by ets.optimize and return a Python float."""
    val = ets._objective_from_params(  # intentionally using internal function for parity
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(y, dtype=jnp.float64),
        n_state,
        error, trend, season,
        opt_crit, n_mse, m,
        opt_alpha, opt_beta, opt_gamma, opt_phi,
        alpha, beta, gamma, phi
    )
    return float(val)

# -------------------------------------------------------------------
# SES: (A, N, N), optimize alpha; fix l0 = y0 via bounds
# -------------------------------------------------------------------

def run_optimize_ses(y: np.ndarray):
    yj = jnp.asarray(y, dtype=jnp.float64)
    n_state = 1  # [l0]
    l0 = float(y[0])

    # Parameter vector p = [alpha] + [states...]
    alpha0 = 0.2
    p0 = jnp.array([alpha0, l0], dtype=jnp.float64)

    # Bounds: alpha in [1e-4, 0.9999]; l0 fixed by lower=upper=l0
    lower = jnp.array([1e-4,  l0], dtype=jnp.float64)
    upper = jnp.array([0.9999, l0], dtype=jnp.float64)

    res = ets.optimize(
        x0=p0,
        y=yj,
        n_state=n_state,
        error=ets.Component.Additive,
        trend=ets.Component.Nothing,
        season=ets.Component.Nothing,
        opt_crit=ets.Criterion.Likelihood,
        n_mse=10,
        m=1,
        opt_alpha=True, opt_beta=False, opt_gamma=False, opt_phi=False,
        alpha=0.0, beta=0.0, gamma=0.0, phi=1.0,
        lower=lower, upper=upper,
        tol_std=1e-8,
        max_iter=800,   # can increase if needed
        adaptive=False,
    )
    return res

def grid_expected_ses(y: np.ndarray):
    """Expected values via grid *using the JAX objective itself* (parity guaranteed)."""
    l0 = float(y[0])
    n_state = 1
    # grid over alpha, keep state tail fixed to l0
    alphas = np.linspace(1e-3, 0.999, 999)
    best = (None, float("inf"))
    for a in alphas:
        p = np.array([a, l0], dtype=np.float64)
        val = jax_objective_from_params(
            p, y, n_state,
            ets.Component.Additive, ets.Component.Nothing, ets.Component.Nothing,
            ets.Criterion.Likelihood, 10, 1,
            True, False, False, False,
            0.0, 0.0, 0.0, 1.0
        )
        if val < best[1]:
            best = (a, val)
    return best  # (alpha*, obj*)

# -------------------------------------------------------------------
# Holt: (A, A, N, phi=1), optimize alpha, beta; fix l0=y0, b0=y1-y0
# -------------------------------------------------------------------

def run_optimize_holt(y: np.ndarray):
    yj = jnp.asarray(y, dtype=jnp.float64)
    n_state = 2  # [l0, b0]
    l0 = float(y[0])
    b0 = float(y[1] - y[0])

    # p = [alpha, beta] + [l0, b0]
    alpha0, beta0 = 0.3, 0.1
    p0 = jnp.array([alpha0, beta0, l0, b0], dtype=jnp.float64)

    # Bounds: alpha,beta in [1e-4, 0.9999]; l0,b0 fixed
    lower = jnp.array([1e-4, 1e-4, l0, b0], dtype=jnp.float64)
    upper = jnp.array([0.9999, 0.9999, l0, b0], dtype=jnp.float64)

    res = ets.optimize(
        x0=p0,
        y=yj,
        n_state=n_state,
        error=ets.Component.Additive,
        trend=ets.Component.Additive,
        season=ets.Component.Nothing,
        opt_crit=ets.Criterion.Likelihood,
        n_mse=10,
        m=1,
        opt_alpha=True, opt_beta=True, opt_gamma=False, opt_phi=False,
        alpha=0.0, beta=0.0, gamma=0.0, phi=1.0,  # phi fixed to 1
        lower=lower, upper=upper,
        tol_std=1e-8,
        max_iter=1200,  # can increase if needed
        adaptive=False,
    )
    return res

def grid_expected_holt(y: np.ndarray):
    """Expected values via 2D grid using the JAX objective itself."""
    l0 = float(y[0])
    b0 = float(y[1] - y[0])
    n_state = 2
    # coarse + refine grids
    def grid(a_center=None, b_center=None, step=0.05, n=101):
        if a_center is None:
            a = np.linspace(0.01, 0.99, 99)
            b = np.linspace(0.01, 0.99, 99)
            return a, b
        lo_a, hi_a = max(0.01, a_center - step), min(0.99, a_center + step)
        lo_b, hi_b = max(0.01, b_center - step), min(0.99, b_center + step)
        return np.linspace(lo_a, hi_a, n), np.linspace(lo_b, hi_b, n)

    def eval_grid(alphas, betas):
        best = (None, None, float("inf"))
        for a in alphas:
            for b in betas:
                p = np.array([a, b, l0, b0], dtype=np.float64)
                val = jax_objective_from_params(
                    p, y, n_state,
                    ets.Component.Additive, ets.Component.Additive, ets.Component.Nothing,
                    ets.Criterion.Likelihood, 10, 1,
                    True, True, False, False,
                    0.0, 0.0, 0.0, 1.0
                )
                if val < best[2]:
                    best = (a, b, val)
        return best

    a1, b1 = grid()
    a_c, b_c, _ = eval_grid(a1, b1)
    a2, b2 = grid(a_c, b_c, step=0.05, n=101)
    return eval_grid(a2, b2)  # (alpha*, beta*, obj*)

# -------------------------------------------------------------------
# Deterministic series
# -------------------------------------------------------------------

def make_series_ses(seed=1234, n=60):
    rng = np.random.default_rng(seed)
    level = 10.0
    noise = rng.normal(0.0, 0.5, size=n)
    return level + noise

def make_series_holt(seed=2024, n=80):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 5.0 + 0.7 * t + rng.normal(0.0, 0.8, size=n)
    return y

# -------------------------------------------------------------------
# Checks (no pytest)
# -------------------------------------------------------------------

def check_ses():
    print("\n=== SES (A,N,N) check ===")
    y = make_series_ses()
    # expected via JAX objective grid
    alpha_star_ref, lik_ref = grid_expected_ses(y)
    # JAX optimize
    res = run_optimize_ses(y)
    alpha_hat = float(res.x[0])
    lik_hat   = float(res.fun)

    print(f"Expected (grid via JAX obj): alpha* = {alpha_star_ref:.4f}, obj* = {lik_ref:.6f}")
    print(f"JAX optimize               : alpha  = {alpha_hat:.4f}, obj  = {lik_hat:.6f}")
    ok_alpha = abs(alpha_hat - alpha_star_ref) < 0.02
    ok_obj   = abs(lik_hat   - lik_ref      ) < 1e-2
    print(f"alpha tolerance pass (<0.02)? {ok_alpha}")
    print(f"obj   tolerance pass (<1e-2)? {ok_obj}")
    return ok_alpha and ok_obj

def check_holt():
    print("\n=== Holt (A,A,N, phi=1) check ===")
    y = make_series_holt()
    # expected via JAX objective grid
    alpha_ref, beta_ref, lik_ref = grid_expected_holt(y)
    # JAX optimize
    res = run_optimize_holt(y)
    alpha_hat = float(res.x[0])
    beta_hat  = float(res.x[1])
    lik_hat   = float(res.fun)

    print(f"Expected (grid via JAX obj): alpha* = {alpha_ref:.4f}, beta* = {beta_ref:.4f}, obj* = {lik_ref:.6f}")
    print(f"JAX optimize               : alpha  = {alpha_hat:.4f}, beta  = {beta_hat:.4f}, obj  = {lik_hat:.6f}")
    ok_alpha = abs(alpha_hat - alpha_ref) < 0.03
    ok_beta  = abs(beta_hat  - beta_ref ) < 0.03
    ok_obj   = abs(lik_hat   - lik_ref  ) < 5e-2
    print(f"alpha tolerance pass (<0.03)? {ok_alpha}")
    print(f"beta  tolerance pass (<0.03)? {ok_beta}")
    print(f"obj   tolerance pass (<5e-2)? {ok_obj}")
    return ok_alpha and ok_beta and ok_obj

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    ok1 = check_ses()
    ok2 = check_holt()
    print("\n=== Summary ===")
    print(f"SES  pass? {ok1}")
    print(f"Holt pass? {ok2}")
    if not (ok1 and ok2):
        print("One or more checks failed.", file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")
