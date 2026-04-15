import pytest
import jax.numpy as jnp

from chronax.utils import ConformalIntervals
from chronax.utils.conformal_methods import (
    add_conformal_distribution_intervals,
    add_conformal_signed_intervals,
    get_conformal_method,
)
from chronax.models.base_forecaster import BaseForecaster


# =========================
# ConformalIntervals class tests
# =========================

def test_default_initialization():
    ci = ConformalIntervals()
    assert ci.n_windows == 2
    assert ci.h == 1
    assert ci.method == "conformal_distribution"

def test_custom_initialization():
    ci = ConformalIntervals(n_windows=10, h=5, method="conformal_signed")
    assert ci.n_windows == 10
    assert ci.h == 5
    assert ci.method == "conformal_signed"

@pytest.mark.parametrize("invalid_windows", [1, 0, -1, -100])
def test_invalid_n_windows_raises_value_error(invalid_windows):
    with pytest.raises(ValueError, match="at least two windows"):
        ConformalIntervals(n_windows=invalid_windows)

def test_invalid_method_raises_value_error():
    with pytest.raises(ValueError, match="method must be one of"):
        ConformalIntervals(method="invalid_method")

def test_boundary_n_windows_succeeds():
    ci = ConformalIntervals(n_windows=2)
    assert ci.n_windows == 2


# =========================
# Interval construction tests
# =========================

def test_conformal_distribution_symmetric():
    """conformal_distribution produces intervals symmetric around the mean."""
    fcst = {"mean": jnp.array([10.0, 20.0, 30.0])}
    cs = jnp.array([[1.0, 2.0, 3.0], [-0.5, -1.0, -1.5]])  # signed scores
    result = BaseForecaster.add_confidence_intervals(fcst, cs, [80], "conformal_distribution")
    assert "lo-80" in result and "hi-80" in result
    assert jnp.allclose(result["mean"] - result["lo-80"], result["hi-80"] - result["mean"])

def test_conformal_signed_allows_asymmetry():
    """conformal_signed can produce asymmetric intervals when errors are skewed."""
    fcst = {"mean": jnp.array([10.0, 20.0])}
    # All positive residuals: model consistently underpredicts
    cs = jnp.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    result = BaseForecaster.add_confidence_intervals(fcst, cs, [80], "conformal_signed")
    assert "lo-80" in result and "hi-80" in result
    # Intervals should be shifted above the mean
    lo_dist = result["mean"] - result["lo-80"]
    hi_dist = result["hi-80"] - result["mean"]
    assert not jnp.allclose(lo_dist, hi_dist)

def test_conformal_signed_keys_match_distribution():
    """Both methods produce the same output keys."""
    fcst_sym = {"mean": jnp.array([5.0, 5.0])}
    fcst_sig = {"mean": jnp.array([5.0, 5.0])}
    cs = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    level = [80, 95]
    r1 = BaseForecaster.add_confidence_intervals(fcst_sym, cs, level, "conformal_distribution")
    r2 = BaseForecaster.add_confidence_intervals(fcst_sig, cs, level, "conformal_signed")
    expected_keys = {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
    assert set(r1.keys()) == expected_keys
    assert set(r2.keys()) == expected_keys

def test_invalid_method_in_add_confidence_intervals():
    fcst = {"mean": jnp.array([1.0])}
    cs = jnp.array([[0.5]])
    with pytest.raises(ValueError):
        BaseForecaster.add_confidence_intervals(fcst, cs, [80], "invalid")


# =========================
# conformal_methods module (pure helpers)
# =========================


def test_get_conformal_method_distribution():
    assert get_conformal_method("conformal_distribution") is add_conformal_distribution_intervals


def test_distribution_and_signed_return_expected_keys_and_shapes():
    fcst_a = {"mean": jnp.array([10.0, 20.0], dtype=jnp.float32)}
    fcst_b = {"mean": jnp.array([10.0, 20.0], dtype=jnp.float32)}
    cs = jnp.array([[1.0, 2.0], [-1.0, -0.5]], dtype=jnp.float32)
    out_a = add_conformal_distribution_intervals(fcst_a, cs, [80, 95])
    out_b = add_conformal_signed_intervals(fcst_b, cs, [80, 95])
    expected_keys = {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
    for out in (out_a, out_b):
        assert set(out.keys()) == expected_keys
        for key in ("lo-80", "hi-80", "lo-95", "hi-95"):
            assert out[key].shape == (2,)
            assert out[key].dtype == jnp.float32