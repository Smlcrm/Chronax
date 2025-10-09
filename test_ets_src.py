## test_ets_src_parity.py
# Parity tests between ets_srcv2 (JAX) and ets_src (NumPy/SciPy).
# - Verifies update/forecast parity (one-step mechanics)
# - Verifies calc (likelihood) parity on a short series
# - Verifies objective parity for AAA undamped/damped and MAM models
# - Adds AAA forward+intervals snapshot test (from your logs) and JAX-vs-NumPy compare
#
# Run as pytest:
#   pytest -q test_ets_src_parity.py
#
# Run directly (no pytest needed):
#   python test_ets_src_parity.py
# or to force pytest from inside:
#   python test_ets_src_parity.py --pytest

import os
os.environ.setdefault("JAX_ENABLE_X64", "True")  # ensure float64 for JAX

from auto_ets import AutoETS
from holt_winters import HoltWinters
import numpy as np
import jax.numpy as jnp
import importlib

# JAX impl
from ets_backend import (
    Component as JComponent,
    Criterion as JCriterion,
    update as j_update,
    forecast as j_forecast,
    calc as j_calc,
    _objective_from_params as j_obj,
    HUGE_N as J_HUGE_N,
    TOL as J_TOL,
)

# NumPy impl
from ets_np_versions_testing.ets_src import (
    Component as NComponent,
    Criterion as NCriterion,
    update as n_update,
    forecast as n_forecast,
    calc as n_calc,
    _objective_function as n_obj,
    HUGE_N as N_HUGE_N,
    TOL as N_TOL,
)

RTOL = 1e-8
ATOL = 1e-8


def _aaa_states(m=4):
    """Initial state vector for AAA: [l0, b0, s1, ..., s_{m-1}]"""
    s = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # m-1=3 (for m=4)
    l0 = 2.0
    b0 = 0.0
    return np.concatenate([[l0, b0], s])  # len = 5 = m+1


def _mam_states(m=4):
    """Initial state vector for MAM: [l0, b0, s1, ..., s_{m-1}] (positive seasonals)."""
    s = np.array([0.88888889, 1.33333333, 1.03703704], dtype=np.float64)
    l0 = 2.0
    b0 = 0.0
    return np.concatenate([[l0, b0], s])  # len = 5 = m+1


def _series_basic_AAA():
    """Short series with m=4 seasonality for AAA."""
    return np.tile(np.array([10.0, 12.0, 13.0, 11.0], dtype=np.float64), 4)  # len 16


def _series_basic_MAM():
    """Short positive series for MAM (m=4)."""
    return np.tile(np.array([5.0, 7.0, 9.0, 6.0], dtype=np.float64), 4)  # len 16


def test_update_parity_AAA():
    m = 4
    l, b = 10.0, 0.5
    s = np.array([0.2, 0.0, -0.1, 0.3], dtype=np.float64)
    old_l, old_b, old_s = 9.8, 0.45, s.copy()
    alpha, beta, gamma, phi = 0.05, 0.005, 0.04, 1.0
    y = 12.0

    # NumPy (mutates seasonal array)
    s_np = s.copy()
    l_np, b_np = n_update(s_np, l, b, old_l, old_b, old_s, m,
                          NComponent.Additive, NComponent.Additive,
                          alpha, beta, gamma, phi, y)

    # JAX (returns seasonal array)
    s_jax = jnp.asarray(s, dtype=jnp.float64)
    old_s_jax = jnp.asarray(old_s, dtype=jnp.float64)
    l_j, b_j, s_j = j_update(
        s_jax, jnp.float64(l), jnp.float64(b), jnp.float64(old_l), jnp.float64(old_b), old_s_jax, m,
        JComponent.Additive, JComponent.Additive, jnp.float64(alpha), jnp.float64(beta),
        jnp.float64(gamma), jnp.float64(phi), jnp.float64(y)
    )

    np.testing.assert_allclose(l_np, float(l_j), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b_np, float(b_j), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(s_np, np.asarray(s_j, dtype=np.float64), rtol=RTOL, atol=ATOL)


def test_forecast_parity_AAA():
    m = 4
    h = 5
    l, b = 10.0, 0.5
    s = np.array([0.2, 0.0, -0.1, 0.3], dtype=np.float64)
    phi = 1.0

    f_np = np.zeros(h, dtype=np.float64)
    n_forecast(f_np, l, b, s.copy(), m, NComponent.Additive, NComponent.Additive, phi, h)

    f_j = jnp.zeros(h, dtype=jnp.float64)
    f_j = j_forecast(f_j, jnp.float64(l), jnp.float64(b), jnp.asarray(s, dtype=jnp.float64),
                     m, JComponent.Additive, JComponent.Additive, jnp.float64(phi), h)

    np.testing.assert_allclose(f_np, np.asarray(f_j, dtype=np.float64), rtol=RTOL, atol=ATOL)


def test_calc_parity_AAA():
    """Compare calc() (likelihood over full series) parity."""
    y = _series_basic_AAA()
    m = 4
    nmse = 3

    # n_states = level + trend + m seasonals = m + 2
    n_states = 1 + 1 + m
    x_head = np.zeros(n_states, dtype=np.float64)
    x_head[0] = 2.0  # l0
    x_head[1] = 0.0  # b0
    x_head[2:] = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float64)

    n = y.size
    x_np = np.zeros(n_states * (n + 1), dtype=np.float64)
    x_np[:n_states] = x_head
    e_np = np.zeros_like(y, dtype=np.float64)
    amse_np = np.zeros(30, dtype=np.float64)

    lik_np = n_calc(x_np.copy(), e_np.copy(), amse_np.copy(), nmse, y.copy(),
                    NComponent.Additive, NComponent.Additive, NComponent.Additive,
                    0.05, 0.005, 0.04, 1.0, m)

    x_j = jnp.zeros_like(jnp.asarray(x_np))
    x_j = x_j.at[:n_states].set(jnp.asarray(x_head))
    e_j = jnp.zeros_like(jnp.asarray(y))
    amse_j = jnp.zeros_like(jnp.asarray(amse_np))
    lik_j = j_calc(x_j, e_j, amse_j, nmse, jnp.asarray(y),
                   JComponent.Additive, JComponent.Additive, JComponent.Additive,
                   jnp.float64(0.05), jnp.float64(0.005), jnp.float64(0.04), jnp.float64(1.0), m)

    np.testing.assert_allclose(float(lik_np), float(lik_j), rtol=RTOL, atol=ATOL)


def test_objective_parity_AAA_undamped():
    """Objective parity for undamped AAA: p=[alpha,beta,gamma,l0,b0,s1..s_{m-1}]"""
    y = _series_basic_AAA()
    m = 4
    nmse = 3
    n_state = m + 1

    alpha, beta, gamma = 0.0137350537, 0.005099, 0.0475905
    states = _aaa_states(m)
    p = np.concatenate([[alpha, beta, gamma], states]).astype(np.float64)

    val_np = n_obj(
        p, y.astype(np.float64), n_state,
        NComponent.Additive, NComponent.Additive, NComponent.Additive,
        NCriterion.Likelihood, nmse, m,
        True, True, True, False,
        np.nan, np.nan, np.nan, 1.0
    )

    val_jx = j_obj(
        jnp.asarray(p), jnp.asarray(y), n_state,
        JComponent.Additive, JComponent.Additive, JComponent.Additive,
        JCriterion.Likelihood, nmse, m,
        True, True, True, False,
        jnp.nan, jnp.nan, jnp.nan, 1.0
    )

    np.testing.assert_allclose(float(val_np), float(val_jx), rtol=RTOL, atol=ATOL)


def test_objective_parity_AAA_damped():
    """Objective parity for damped AAA: p=[alpha,beta,gamma,phi,l0,b0,s1..s_{m-1}]"""
    y = _series_basic_AAA()
    m = 4
    nmse = 3
    n_state = m + 1

    alpha, beta, gamma, phi = 0.051, 0.00517, 0.0476, 0.978
    states = _aaa_states(m)
    p = np.concatenate([[alpha, beta, gamma, phi], states]).astype(np.float64)

    val_np = n_obj(
        p, y.astype(np.float64), n_state,
        NComponent.Additive, NComponent.Additive, NComponent.Additive,
        NCriterion.Likelihood, nmse, m,
        True, True, True, True,
        np.nan, np.nan, np.nan, np.nan
    )

    val_jx = j_obj(
        jnp.asarray(p), jnp.asarray(y), n_state,
        JComponent.Additive, JComponent.Additive, JComponent.Additive,
        JCriterion.Likelihood, nmse, m,
        True, True, True, True,
        jnp.nan, jnp.nan, jnp.nan, jnp.nan
    )

    np.testing.assert_allclose(float(val_np), float(val_jx), rtol=RTOL, atol=ATOL)


def test_objective_parity_MAM_undamped():
    """Objective parity for undamped MAM."""
    y = _series_basic_MAM()
    m = 4
    nmse = 3
    n_state = m + 1

    alpha, beta, gamma = 0.051, 0.00515, 0.0477
    states = _mam_states(m)
    p = np.concatenate([[alpha, beta, gamma], states]).astype(np.float64)

    val_np = n_obj(
        p, y.astype(np.float64), n_state,
        NComponent.Multiplicative, NComponent.Additive, NComponent.Multiplicative,
        NCriterion.Likelihood, nmse, m,
        True, True, True, False,
        np.nan, np.nan, np.nan, 1.0
    )

    val_jx = j_obj(
        jnp.asarray(p), jnp.asarray(y), n_state,
        JComponent.Multiplicative, JComponent.Additive, JComponent.Multiplicative,
        JCriterion.Likelihood, nmse, m,
        True, True, True, False,
        jnp.nan, jnp.nan, jnp.nan, 1.0
    )

    np.testing.assert_allclose(float(val_np), float(val_jx), rtol=RTOL, atol=ATOL)


def test_forecast_parity_MAM():
    """Direct multi-step forecast parity under MAM with fixed states."""
    m = 4
    h = 6
    l, b = 7.0, 0.2
    s = np.array([0.9, 1.1, 1.0, 1.0], dtype=np.float64)
    phi = 1.0

    f_np = np.zeros(h, dtype=np.float64)
    n_forecast(f_np, l, b, s.copy(), m, NComponent.Additive, NComponent.Multiplicative, phi, h)

    f_j = jnp.zeros(h, dtype=jnp.float64)
    f_j = j_forecast(f_j, jnp.float64(l), jnp.float64(b), jnp.asarray(s, dtype=jnp.float64),
                     m, JComponent.Additive, JComponent.Multiplicative, jnp.float64(phi), h)

    np.testing.assert_allclose(f_np, np.asarray(f_j, dtype=np.float64), rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------
# NEW: AAA forward+intervals tests (snapshot + cross-backend comparison)
# --------------------------------------------------------------------

# Expected arrays pulled from your logs:
EXP_JAX = {
    "lo-80": np.array([0.35277287, 1.39978994, 2.44612750, 1.49179030]),
    "mean":  np.array([1.83791211, 2.88519256, 3.93195532, 2.97824358]),
    "hi-80": np.array([3.32305135, 4.37059519, 5.41778314, 4.46469687]),
    "lo-95": np.array([-0.43341236,  0.61346529,  1.65957776,  0.70490947]),
    "hi-95": np.array([ 4.10923657,  5.15691983,  6.20433288,  5.25157770]),
}

EXP_NUMPY = {
    "lo-80": np.array([0.76995194, 1.80027210, 2.83021160, 1.85978710]),
    "mean":  np.array([2.00599201, 3.03829590, 4.07059383, 3.10293306]),
    "hi-80": np.array([3.24203200, 4.27632000, 5.31097600, 4.34607930]),
    "lo-95": np.array([0.11563206, 1.14490200, 2.17359300, 1.20170530]),
    "hi-95": np.array([3.89635180, 4.93169000, 5.96759460, 5.00416100]),
}

_KEYS = ("lo-80", "mean", "hi-80", "lo-95", "hi-95")


def _load_class(dotted: str):
    """Load a class from 'package.module:Class' or 'package.module.Class'."""
    if ":" in dotted:
        mod, cls = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        mod, cls = ".".join(parts[:-1]), parts[-1]
    m = importlib.import_module(mod)
    return getattr(m, cls)


def _maybe_import(cls_path):
    try:
        return _load_class(cls_path)
    except Exception:
        return None


def _make_model(HoltWintersClass):
    # be robust to slightly different ctor signatures across backends
    try:
        return HoltWintersClass(season_length=4, error_type="A", trend_type="A", season_type="A")
    except TypeError:
        return HoltWintersClass(season_length=4, error_type="A")


def _forward_dict(hw_cls):
    y_fit = np.asarray([1.0, 2.0, 3.0, 2.0] * 4, dtype=np.float64)
    y_new = np.asarray([2.0, 3.0, 4.0, 3.0] * 4, dtype=np.float64)
    m = _make_model(hw_cls)
    m.fit(y_fit)
    out = m.forward(y=y_new, h=4, level=[80, 95], fitted=False)
    return {k: np.asarray(out[k], dtype=np.float64) for k in _KEYS}


def _print_diff(label, a, b):
    diff = a - b
    print(f"\n--- {label} ---")
    print("A:", a)
    print("B:", b)
    print("abs diff:", np.abs(diff))
    print("max abs diff:", np.max(np.abs(diff)))


def test_forward_intervals_snapshot_from_logs():
    """
    Snapshot check against your log values.

    Choose which snapshot via env:
      ETS_SNAPSHOT=jax     -> compares to EXP_JAX
      ETS_SNAPSHOT=numpy   -> compares to EXP_NUMPY

    Choose which class via env (defaults to your JAX wrapper):
      ETS_CLASS="auto_ets.HoltWinters"
    """
    target = os.getenv("ETS_SNAPSHOT", "jax").lower()
    exp = EXP_JAX if target == "jax" else EXP_NUMPY

    cls_path = os.getenv("ETS_CLASS", "holt_winters.HoltWinters")
    HW = _load_class(cls_path)

    out = _forward_dict(HW)

    # print like your original test
    for lv in [80, 95]:
        print(out[f"lo-{lv}"])
        print(out["mean"], "mean")
        print(out[f"hi-{lv}"])

    for k in _KEYS:
        np.testing.assert_allclose(out[k], exp[k], rtol=1e-6, atol=1e-6, err_msg=f"Mismatch for {k} ({target} snapshot)")


def test_forward_intervals_jax_vs_numpy_if_available():
    """
    If both backends are importable, compare them head-to-head on the AAA forward case.
    This will currently FAIL for you (by design) to surface the divergence.
    If either backend class can't be imported, the test is a no-op (passes).
    """
    jax_path = os.getenv("ETS_JAX_CLASS", "auto_ets.HoltWinters")
    numpy_path = os.getenv("ETS_NUMPY_CLASS", "ets_np_versions_testing.auto_ets_np.HoltWinters")

    HW_JAX = _maybe_import(jax_path)
    HW_NP  = _maybe_import(numpy_path)

    if HW_JAX is None or HW_NP is None:
        # No-op when run without both wrappers present
        print(f"Skipping head-to-head; importable? JAX={HW_JAX is not None}, NumPy={HW_NP is not None}")
        return

    out_jax = _forward_dict(HW_JAX)
    out_np  = _forward_dict(HW_NP)

    # print like your original test
    for lv in [80, 95]:
        print(out_jax[f"lo-{lv}"]); print(out_jax["mean"], "mean"); print(out_jax[f"hi-{lv}"])
    for lv in [80, 95]:
        print(out_np[f"lo-{lv}"]); print(out_np["mean"], "mean"); print(out_np[f"hi-{lv}"])

    for k in _KEYS:
        if not np.allclose(out_jax[k], out_np[k], rtol=1e-6, atol=1e-6):
            _print_diff(k + " (JAX - NumPy)", out_jax[k], out_np[k])
    for k in _KEYS:
        np.testing.assert_allclose(out_jax[k], out_np[k], rtol=1e-6, atol=1e-6, err_msg=f"JAX vs NumPy mismatch in {k}")


# -------------------------------
# Script harness to run tests directly
# -------------------------------
def _run(func):
    name = func.__name__
    try:
        func()
        print(f"{name}: OK")
        return True
    except AssertionError as e:
        print(f"{name}: FAIL — {e}")
        return False
    except Exception as e:
        print(f"{name}: ERROR — {type(e).__name__}: {e}")
        return False


def run_all_tests():
    tests = [
        test_update_parity_AAA,
        test_forecast_parity_AAA,
        test_calc_parity_AAA,
        test_objective_parity_AAA_undamped,
        test_objective_parity_AAA_damped,
        test_objective_parity_MAM_undamped,
        test_forecast_parity_MAM,
        # NEW:
        test_forward_intervals_snapshot_from_logs,
        test_forward_intervals_jax_vs_numpy_if_available,
    ]
    ok = 0
    for t in tests:
        ok += int(_run(t))
    total = len(tests)
    print(f"\nSummary: {ok}/{total} passed")
    return ok == total


def run_smoke():
    """Small smoke demo (same as before) to print objective diffs."""
    def _print(name, val):
        print(f"{name}: {val:.12f}" if isinstance(val, (float, np.floating)) else f"{name}: {val}")

    print("\nJAX/NumPy ETS parity quick run\n")

    # AAA undamped objective
    y = _series_basic_AAA()
    m = 4
    nmse = 3
    n_state = m + 1
    alpha, beta, gamma = 0.0137, 0.005099, 0.04759
    p_aaa = np.concatenate([[alpha, beta, gamma], _aaa_states(m)]).astype(np.float64)

    v_np = n_obj(p_aaa, y.astype(np.float64), n_state,
                 NComponent.Additive, NComponent.Additive, NComponent.Additive,
                 NCriterion.Likelihood, nmse, m,
                 True, True, True, False, np.nan, np.nan, np.nan, 1.0)
    v_jx = j_obj(jnp.asarray(p_aaa), jnp.asarray(y), n_state,
                 JComponent.Additive, JComponent.Additive, JComponent.Additive,
                 JCriterion.Likelihood, nmse, m,
                 True, True, True, False, jnp.nan, jnp.nan, jnp.nan, 1.0)

    _print("AAA undamped - NumPy objective", float(v_np))
    _print("AAA undamped - JAX  objective", float(v_jx))
    print("abs diff:", abs(float(v_np) - float(v_jx)))

    # MAM undamped objective
    y2 = _series_basic_MAM()
    p_mam = np.concatenate([[0.051, 0.00515, 0.0477], _mam_states(m)]).astype(np.float64)

    v2_np = n_obj(p_mam, y2.astype(np.float64), n_state,
                  NComponent.Multiplicative, NComponent.Additive, NComponent.Multiplicative,
                  NCriterion.Likelihood, nmse, m,
                  True, True, True, False, np.nan, np.nan, np.nan, 1.0)
    v2_jx = j_obj(jnp.asarray(p_mam), jnp.asarray(y2), n_state,
                  JComponent.Multiplicative, JComponent.Additive, JComponent.Multiplicative,
                  JCriterion.Likelihood, nmse, m,
                  True, True, True, False, jnp.nan, jnp.nan, jnp.nan, 1.0)

    _print("MAM undamped - NumPy objective", float(v2_np))
    _print("MAM undamped - JAX  objective", float(v2_jx))
    print("abs diff:", abs(float(v2_np) - float(v2_jx)))
    print("\nDone.\n")


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from jaxopt import ScipyMinimize

    # Define the function to minimize
    def rosenbrock(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

    # Create the optimizer instance, specifying Nelder-Mead
    nelder_mead = ScipyMinimize(method='Nelder-Mead', fun=rosenbrock)

    # Define the initial guess
    x0 = jnp.array([1.3, 0.7, 0.8, 1.9, 1.2])

    # Run the optimization
    params, state = nelder_mead.run(init_params=x0)

    # Print the results
    print("Found minimum at:", params)
    import argparse, sys

    parser = argparse.ArgumentParser(description="Run ETS parity tests.")
    parser.add_argument("--pytest", action="store_true",
                        help="Run the file via pytest.main() instead of the built-in runner.")
    parser.add_argument("--no-smoke", action="store_true",
                        help="Skip the smoke demo.")
    args = parser.parse_args()

    if args.pytest:
        try:
            import pytest
        except Exception as e:
            print("pytest not available:", e)
            sys.exit(2)
        # Run this file through pytest with quiet output
        rc = pytest.main([__file__, "-q"])
        sys.exit(rc)

    print("Running built-in test runner...\n")
    all_ok = run_all_tests()
    if not args.no_smoke:
        run_smoke()
    sys.exit(0 if all_ok else 1)
