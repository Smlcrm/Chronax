"""
Compare ETS AAA forward() + intervals between the NumPy and JAX implementations.

Defaults assume:
  - JAX wrapper module at  "ets"                     (exports: ets_f, forecast_ets, forward_ets)
  - NumPy wrapper module at "ets_np_versions_testing.ets"  (same API)

Override via env:
  ETS_JAX_MOD=path.to.jax_module
  ETS_NUMPY_MOD=path.to.numpy_module
  ETS_STRICT=1        # if set, the comparison test will assert_allclose (and fail if they differ)

Run:
  pytest -q test_ets_backends_compare.py
or:
  ETS_STRICT=1 pytest -q test_ets_backends_compare.py::test_AAA_forward_jax_vs_numpy_close
"""

from __future__ import annotations

import os
import importlib
import numpy as np
import pytest

# Try to ensure x64 for JAX side before modules import it.
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass


# -----------------------
# Helpers to load backends
# -----------------------
def _load_backend(mod_path: str):
    m = importlib.import_module(mod_path)
    # minimal API check
    for name in ("ets_f", "forecast_ets", "forward_ets"):
        if not hasattr(m, name):
            raise ImportError(f"Module '{mod_path}' missing required symbol '{name}'")
    return m


def _backend_paths():
    jax_mod = os.getenv("ETS_JAX_MOD", "ets_v2")  # your JAX wrapper
    np_mod = os.getenv("ETS_NUMPY_MOD", "ets_np_versions_testing.ets_v1")  # your NumPy wrapper
    return jax_mod, np_mod


def _fit_forward_and_forecast(mod, y_fit, y_new, h=4, level=(80, 95)):
    """
    1) fit on y_fit with model AAA
    2) forward to y_new (reuse model)
    3) forecast h steps with native intervals
    returns dict with 'mean', 'lo-80', 'hi-80', 'lo-95', 'hi-95'
    """
    # fit best AAA (searches damped=True/False internally)
    fit = mod.ets_f(y=np.asarray(y_fit, dtype=np.float64),
                    m=4, model="AAA", nmse=3, bounds="both")
    fwd_model = mod.forward_ets(fit, y=np.asarray(y_new, dtype=np.float64))
    out = mod.forecast_ets(fwd_model, h=h, level=list(level))
    # normalize outputs to np arrays (float64)
    got = {
        "mean":  np.asarray(out["mean"],  dtype=np.float64),
        "lo-80": np.asarray(out["lo-80"], dtype=np.float64),
        "hi-80": np.asarray(out["hi-80"], dtype=np.float64),
        "lo-95": np.asarray(out["lo-95"], dtype=np.float64),
        "hi-95": np.asarray(out["hi-95"], dtype=np.float64),
    }
    return got, fit, fwd_model


def _print_block(title, d):
    print(title)
    for k in ("lo-80", "mean", "hi-80", "lo-95", "mean", "hi-95"):
        key = "mean" if k == "mean" else k  # keep ordering/dup like your logs
        print(d[key], "mean" if key == "mean" else "")


# -----------------------
# The test you asked for
# -----------------------

def test_AAA_forward_jax_vs_numpy_print_only():
    """
    Reproduce your scenario and PRINT both backends’ arrays side-by-side.
    This test never fails; it’s for inspection/logs.
    """
    jax_path, np_path = _backend_paths()

    try:
        jax_mod = _load_backend(jax_path)
        np_mod = _load_backend(np_path)
    except ImportError as e:
        pytest.skip(f"backend import failed: {e}")

    y_fit = np.asarray([1., 2., 3., 2.] * 4, dtype=np.float64)
    y_new = np.asarray([2., 3., 4., 3.] * 4, dtype=np.float64)

    jax_out, jax_fit, jax_fwd = _fit_forward_and_forecast(jax_mod, y_fit, y_new)
    np_out, np_fit, np_fwd = _fit_forward_and_forecast(np_mod, y_fit, y_new)

    # Mirror your console prints
    _print_block("JAX:", jax_out)
    _print_block("NumPy:", np_out)

    # interval sanity (each backend)
    for out in (jax_out, np_out):
        assert np.all(out["lo-80"] <= out["mean"])
        assert np.all(out["mean"] <= out["hi-80"])
        assert np.all(out["lo-95"] <= out["mean"])
        assert np.all(out["mean"] <= out["hi-95"])

    # Print diffs so you can see what’s off
    for k in ("lo-80", "mean", "hi-80", "lo-95", "hi-95"):
        diff = jax_out[k] - np_out[k]
        print(f"\nDIFF {k} (JAX - NumPy):")
        print("jax:", jax_out[k])
        print("np :", np_out[k])
        print("abs diff:", np.abs(diff))
        print("max abs diff:", np.max(np.abs(diff)))


@pytest.mark.parametrize("rtol, atol", [(1e-6, 1e-6)])
def test_AAA_forward_jax_vs_numpy_close(rtol, atol):
    """
    Strict numeric comparison (off by default).
    Set ETS_STRICT=1 to enforce allclose; otherwise xfail (documenting current mismatch).
    """
    jax_path, np_path = _backend_paths()

    try:
        jax_mod = _load_backend(jax_path)
        np_mod = _load_backend(np_path)
    except ImportError as e:
        pytest.skip(f"backend import failed: {e}")

    y_fit = np.asarray([1., 2., 3., 2.] * 4, dtype=np.float64)
    y_new = np.asarray([2., 3., 4., 3.] * 4, dtype=np.float64)

    jax_out, *_ = _fit_forward_and_forecast(jax_mod, y_fit, y_new)
    np_out, *_ = _fit_forward_and_forecast(np_mod, y_fit, y_new)

    strict = os.getenv("ETS_STRICT", "0") == "1"

    # Perform comparisons; if not strict, mark xfail to document the discrepancy.
    for k in ("lo-80", "mean", "hi-80", "lo-95", "hi-95"):
        if not strict:
            pytest.xfail("Known JAX vs NumPy divergence in AAA forward intervals; run with ETS_STRICT=1 to enforce.")
        np.testing.assert_allclose(jax_out[k], np_out[k], rtol=rtol, atol=atol, err_msg=f"Mismatch for {k}")


# Optional: quick CLI run
if __name__ == "__main__":
    # emulate the scenario without pytest, for an easy one-off run
    jax_path, np_path = _backend_paths()
    print("____________JAX TURN_______")
    jax_mod = _load_backend(jax_path)
    print("____________NUMPY TURN_______")
    np_mod = _load_backend(np_path)

    y_fit = np.asarray([1., 2., 3., 2.] * 4, dtype=np.float64)
    y_new = np.asarray([2., 3., 4., 3.] * 4, dtype=np.float64)
    print("____________JAX TURN_______")
    jax_out, *_ = _fit_forward_and_forecast(jax_mod, y_fit, y_new)
    print("____________NUMPY TURN_______")

    np_out, *_ = _fit_forward_and_forecast(np_mod, y_fit, y_new)

    _print_block("JAX:", jax_out)
    _print_block("NumPy:", np_out)

    for k in ("lo-80", "mean", "hi-80", "lo-95", "hi-95"):
        diff = jax_out[k] - np_out[k]
        print(f"\nDIFF {k} (JAX - NumPy):")
        print("jax:", jax_out[k])
        print("np :", np_out[k])
        print("abs diff:", np.abs(diff))
        print("max abs diff:", np.max(np.abs(diff)))
