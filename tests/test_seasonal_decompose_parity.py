"""statsmodels-parity guard for the JAX ``utils.seasonal_decompose``.

ETS ``initstate`` seeds its seasonal state from ``utils.seasonal_decompose``
(NaN-aware centred MA, edge rows masked out of the per-period average). This
guard pins that decomposition to statsmodels' classical ``seasonal_decompose``
so the ETS seasonal seed cannot silently drift back toward the old
edge-biased ``mode='same'`` behaviour (which diverged 0.3-2.5x on additive
seasonals at large period).

statsmodels is an optional parity dependency — skipped when absent, matching
the other parity tests.
"""
import numpy as np
import jax.numpy as jnp
import pytest

sm_seasonal = pytest.importorskip(
    "statsmodels.tsa.seasonal"
).seasonal_decompose

from chronax.utils.utils import seasonal_decompose as cx_decompose


def _series(m, n_periods, mult, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(m * n_periods, dtype=np.float64)
    seas = 10.0 * np.sin(2 * np.pi * (t % m) / m) + 3.0 * ((t % m) - m / 2)
    trend = 100.0 + 0.5 * t
    noise = rng.normal(0, 1.0, t.shape)
    if mult:
        return trend * (1.0 + 0.1 * seas / seas.std()) * (1.0 + 0.02 * noise)
    return trend + seas + noise


@pytest.mark.parametrize("m", [7, 12, 24, 52])
@pytest.mark.parametrize("mult", [False, True])
def test_seasonal_decompose_matches_statsmodels(m, mult):
    """The JAX seasonal component equals statsmodels' at classical-MA parity."""
    y = _series(m, 6, mult, seed=m * 100 + int(mult))
    model = "multiplicative" if mult else "additive"

    sm = np.asarray(sm_seasonal(y, model=model, period=m, two_sided=True).seasonal)
    cx = np.asarray(cx_decompose(jnp.asarray(y), model=model, period=m)["seasonal"])

    scale = np.abs(sm).mean() + 1e-12
    assert np.max(np.abs(cx - sm)) / scale < 1e-6, (
        f"m={m} mult={mult}: max rel dev "
        f"{np.max(np.abs(cx - sm)) / scale:.2e} exceeds 1e-6"
    )


@pytest.mark.parametrize("m", [7, 12])
def test_ets_initstate_seasonal_seed_tracks_statsmodels(m):
    """ETS seeds init_seas from seasonal[1:m][::-1]; that slice tracks statsmodels."""
    y = _series(m, 6, mult=False, seed=m)
    sm = np.asarray(sm_seasonal(y, model="additive", period=m, two_sided=True).seasonal)
    cx = np.asarray(cx_decompose(jnp.asarray(y), model="additive", period=m)["seasonal"])
    sm_seed = sm[1:m][::-1]
    cx_seed = cx[1:m][::-1]
    scale = np.abs(sm_seed).mean() + 1e-12
    assert np.max(np.abs(cx_seed - sm_seed)) / scale < 1e-6
