"""BaseForecaster contract checks for classical models that have no per-model test file.

``BaseForecaster.conformity_scores`` vmaps ``forecast()`` over the CV windows, so every
path reachable from ``forecast()`` must trace under ``jax.vmap``. These tests cover that
path, the ``level`` branch (which the CV path never exercises), pickling, repeat-call
compilation, and the MFLES numerical kernels that replace ``jnp.median`` and the plain
LASSO gradient.
"""

import io
import logging
import pickle
import re

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

import chronax.models as M
from chronax.models.mfles import mfles as cxm
from chronax.utils.conformal_intervals import ConformalIntervals

H = 4
N_WINDOWS = 2

MODELS = {
    "GARCH": {},
    "AutoCES": dict(season_length=12),
    "MFLES": {},
    "STL": dict(period=12),
    "Naive": {},
    "RandomWalkWithDrift": {},
}


def _y():
    t = np.arange(60, dtype=np.float64)
    return jnp.asarray(10.0 + 0.7 * t + 3.0 * np.sin(t / 3.0) + np.cos(7.0 * t))


def _make(name):
    ci = ConformalIntervals(n_windows=N_WINDOWS, h=H, method="conformal_distribution")
    return getattr(M, name)(conformal_params=ci, **MODELS[name])


@pytest.mark.parametrize("name", sorted(MODELS))
def test_conformity_scores_under_vmap(name):
    cs = _make(name).new().conformity_scores(_y())
    assert cs.shape == (N_WINDOWS, H)
    assert bool(jnp.all(jnp.isfinite(cs)))
    # A forecast() that ignores its y argument gives identical scores in every window.
    assert not bool(jnp.allclose(cs[0], cs[1]))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_predict_level_twice_then_pickle(name):
    m = _make(name)
    m.fit(_y())
    # The second call is where tracers left on the estimator by the CV vmap surface.
    for p in (m.predict(h=H, level=[80]), m.predict(h=H, level=[80])):
        assert {"mean", "lo-80", "hi-80"} <= set(p)
        assert bool(jnp.all(p["lo-80"] <= p["hi-80"]))
    p = pickle.loads(pickle.dumps(m)).predict(h=H, level=[80])
    assert {"lo-80", "hi-80"} <= set(p)


def test_ses_stateless_forecast_emits_intervals():
    ci = ConformalIntervals(n_windows=N_WINDOWS, h=H)
    m = M.SimpleExponentialSmoothing(alpha=0.5, conformal_params=ci)
    p = m.forecast(y=_y(), h=H, level=[80])
    assert {"mean", "lo-80", "hi-80"} <= set(p)
    lo, hi = np.asarray(p["lo-80"]), np.asarray(p["hi-80"])
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(hi >= lo)


_COMPILE_LOG_RE = re.compile(r"Finished XLA compilation")


def _count_compiles(fn):
    """Run fn under jax.log_compiles and return (result, number of XLA compilations)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    loggers = [logging.getLogger("jax._src.dispatch"),
               logging.getLogger("jax._src.interpreters.pxla")]
    with jax.log_compiles(True):
        for lg in loggers:
            lg.addHandler(handler)
        try:
            out = fn()
            jax.block_until_ready(out)
        finally:
            for lg in loggers:
                lg.removeHandler(handler)
    return out, len(_COMPILE_LOG_RE.findall(buf.getvalue()))


def test_count_compiles_detects_a_compile():
    """Keeps the no-recompile tests below from passing vacuously."""
    def fresh_scan():
        return lax.scan(lambda c, x: (c + x, c), jnp.asarray(0.0), jnp.arange(13.0))[0]

    fresh_scan()
    assert _count_compiles(fresh_scan)[1] >= 1


@pytest.mark.parametrize("name", ["AutoCES", "GARCH"])
def test_repeat_forecast_and_refit_do_not_recompile(name):
    y = _y()
    model = getattr(M, name)(**MODELS[name])
    model.forecast(h=H, y=y)
    model.forecast(h=H, y=y)
    _, n = _count_compiles(lambda: model.forecast(h=H, y=y)["mean"])
    assert n == 0, f"{name}: repeat forecast compiled {n} time(s)"

    model.fit(y)
    jax.block_until_ready(model.predict(h=H)["mean"])

    def refit():
        model.fit(y)
        return model.predict(h=H)["mean"]

    _, n = _count_compiles(refit)
    assert n == 0, f"{name}: refit of a fitted instance compiled {n} time(s)"


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("n", [31, 32])
def test_mfles_bisect_median_matches_jnp_median(dtype, n):
    rng = np.random.RandomState(n)
    A = jnp.asarray(rng.randn(n, n) * 10.0, dtype=dtype)
    A = A.at[0].set(A[0, 0])  # a row of ties
    A = A.at[1].set(-A[1])
    got = jax.jit(cxm._bisect_median_axis1)(A)
    want = jax.jit(lambda a: jnp.median(a, axis=1))(A)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_mfles_siegel_fits_a_line_through_outliers():
    x = jnp.arange(40, dtype=jnp.float64)
    y = 2.0 + 0.5 * x
    y_out = y.at[jnp.array([3, 11, 25, 33])].add(jnp.array([40.0, -35.0, 50.0, -60.0]))
    fitted = cxm._siegel_repeated_medians(x, y_out)
    np.testing.assert_allclose(np.asarray(fitted), np.asarray(y), atol=1e-8)


def test_mfles_lasso_gram_form_matches_plain_gradient():
    rng = np.random.RandomState(0)
    X = jnp.asarray(rng.randn(80, 9))
    y = jnp.asarray(rng.randn(80))
    alpha, maxiter = 0.1, 150
    step = 1.0 / jnp.linalg.norm(X, ord=2) ** 2

    def reference():
        b = jnp.zeros(X.shape[1])
        for _ in range(maxiter):
            z = b - step * (X.T @ (X @ b - y))
            b = jnp.sign(z) * jnp.maximum(jnp.abs(z) - step * alpha, 0.0)
        return b

    want = np.asarray(reference())
    for G in (None, X.T @ X):
        got = cxm._lasso_ista_with_step(X, y, alpha, step, maxiter=maxiter, G=G)
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-10, atol=1e-12)
