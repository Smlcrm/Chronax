
import jax
import jax.numpy as jnp
from chronax.utils import ConformalIntervals
from chronax.models import SeasonalExponentialSmoothing, _seasonal_exponential_smoothing

def test_seasonal_exponential_smoothing():
    y = jnp.arange(36.0)

    pi = ConformalIntervals(h=12, n_windows=2)
    model = SeasonalExponentialSmoothing(season_length=12, alpha=0.5, conformal_params=pi)
    fitted_model = model.fit(y)

    result = fitted_model.predict(h=12, level=(60,75))
    forecast = fitted_model.forecast(y, h=12, level=[80, 95])
    
    assert "mean" in result, "Missing mean forecast"

    assert len(result["mean"]) == 12, "Forecast length mismatch"

    for lvl in [60, 75]:
        if f"lo-{lvl}" in result:
            assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
        else:
            print(f"Warning: Interval {lvl}% not computed due to missing `conformal_params`")

    for lvl in [80, 95]:
        assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
        assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
        assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

def test_numerical_consistency():
    """Same input -> same output (determinism)."""
    y = jnp.arange(72.0)
    for season_length in [12, 24]:
        for alpha in [0.2, 0.5, 0.8]:
            mod1 = _seasonal_exponential_smoothing(
                y=y, h=12, fitted=True, season_length=season_length, alpha=alpha
            )
            mod2 = _seasonal_exponential_smoothing(
                y=y, h=12, fitted=True, season_length=season_length, alpha=alpha
            )
            assert jnp.allclose(mod1["mean"], mod2["mean"]), f"Mismatch for s={season_length}, a={alpha}"
            assert jnp.allclose(mod1["fitted"], mod2["fitted"], equal_nan=True), f"Fitted mismatch for s={season_length}, a={alpha}"


def test_core_kernel_output_shapes():
    """Core kernel returns correct shapes and no NaN when data sufficient."""
    y = jnp.arange(48.0)
    mod = _seasonal_exponential_smoothing(
        y=y, h=24, fitted=True, season_length=12, alpha=0.5
    )
    assert mod["mean"].shape == (24,)
    assert mod["fitted"].shape == y.shape
    assert not jnp.any(jnp.isnan(mod["mean"])), "Mean forecast should have no NaN"
    assert jnp.any(jnp.isnan(mod["fitted"])), "First fitted value is NaN by design"


def test_short_series_returns_nan_mean():
    """Short series (< season_length) returns NaN mean."""
    y = jnp.arange(8.0)
    mod = _seasonal_exponential_smoothing(
        y=y, h=4, fitted=False, season_length=12, alpha=0.5
    )
    assert jnp.all(jnp.isnan(mod["mean"])), "Short series should produce NaN forecast"


if __name__ == "__main__":
    test_seasonal_exponential_smoothing()
    test_numerical_consistency()
    test_core_kernel_output_shapes()
    test_short_series_returns_nan_mean()
    print("All tests passed!")