import numpy as np
import jax.numpy as jnp

from chronax.models.loess import loess_window_jump
from chronax.models import STL


def test_window_1_deg_1_is_identity():
    # window=1 has no neighbours: the local linear fit must return the point
    # itself, not identically zero (the degenerate 0/eps result).
    y = jnp.asarray([3.0, 5.0, 2.0, 8.0, 1.0, 4.0])
    out = np.asarray(loess_window_jump(y, window=1, deg=1))
    np.testing.assert_allclose(out, np.asarray(y), rtol=1e-10)


def test_robust_reweighting_uses_per_observation_weights():
    # A single gross outlier must be down-weighted so the robust smooth stays
    # near the clean neighbours there, not collapse toward zero.
    rng = np.random.default_rng(0)
    n = 60
    clean = 10.0 + np.sin(2 * np.pi * np.arange(n) / 20)
    y = clean.copy()
    y[30] = 1000.0  # gross outlier
    yj = jnp.asarray(y)
    robust = np.asarray(loess_window_jump(yj, window=11, deg=1, robust_outer=3))
    # At the outlier the robust fit tracks the local level (~10), far from
    # both the outlier and zero.
    assert 5.0 < robust[30] < 20.0, f"robust fit at outlier = {robust[30]:.2f}"


def test_stl_predict_in_sample_level_shapes():
    rng = np.random.default_rng(1)
    n = 120
    t = np.arange(n)
    y = jnp.asarray(50 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n))
    m = STL(period=12).fit(y)
    out = m.predict_in_sample(level=[95])
    assert np.asarray(out["fitted-lo-95"]).shape == (n,)
    assert np.all(np.asarray(out["fitted-lo-95"]) < np.asarray(out["mean"]))


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main(["-q", __file__]))
