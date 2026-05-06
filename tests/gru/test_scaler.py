"""Tests for chronax.models.gru.gru_scaler — Scaler protocol + RobustScaler."""
import jax.numpy as jnp
import numpy as np

from chronax.models.gru.gru_scaler import RobustScaler, Scaler


def test_robust_scaler_implements_scaler_protocol():
    sc = RobustScaler()
    assert isinstance(sc, Scaler)


def test_robust_stats_returns_median_and_mad_along_time_axis():
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [10.0, 20.0, 30.0, 40.0, 50.0]])
    sc = RobustScaler()
    shift, scale = sc.stats(x, axis=1)
    assert shift.shape == (2, 1)
    assert scale.shape == (2, 1)
    np.testing.assert_allclose(shift[:, 0], jnp.array([3.0, 30.0]))
    # MAD of [1..5] = median(|x - 3|) = median([2,1,0,1,2]) = 1
    np.testing.assert_allclose(scale[:, 0] - 1e-6, jnp.array([1.0, 10.0]), rtol=1e-6)


def test_robust_transform_inverse_roundtrip():
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sc = RobustScaler()
    shift, scale = sc.stats(x, axis=1)
    z = sc.transform(x, shift, scale)
    x_back = sc.inverse(z, shift, scale)
    np.testing.assert_allclose(x_back, x, rtol=1e-6)


def test_robust_stats_mad_zero_falls_back_to_std_times_constant():
    # Median=1, |dev|=[0,0,1,0,0], MAD=median(...)=0 -> fallback path triggers.
    x = jnp.array([[1.0, 1.0, 2.0, 1.0, 1.0]])
    sc = RobustScaler()
    shift, scale = sc.stats(x, axis=1)
    raw_mad = float(jnp.median(jnp.abs(x[0] - 1.0)))
    assert raw_mad == 0.0, "test setup invalid"  # fallback IS exercised
    std = float(jnp.std(x[0]))
    expected_fallback = std * 0.6744897501960817 + 1e-6
    assert abs(float(scale[0, 0]) - expected_fallback) < 1e-6


def test_robust_stats_mask_excludes_future_steps():
    x = jnp.array([[1.0, 2.0, 3.0, 100.0, 200.0]])
    mask = jnp.array([[True, True, True, False, False]])
    sc = RobustScaler()
    shift, _ = sc.stats(x, axis=1, mask=mask)
    assert shift[0, 0] == 2.0  # median of input portion only
