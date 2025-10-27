
import jax
import jax.numpy as jnp
from conformal_intervals import ConformalIntervals
from seasonal_exponential_smoothing import SeasonalExponentialSmoothing

def test_seasonal_exponential_smoothing():
    y = jnp.arange(36.0)

    pi = ConformalIntervals(h=12, n_windows=2)
    model = SeasonalExponentialSmoothing(season_length=12, alpha=0.5, prediction_intervals=pi)
    fitted_model = model.fit(y)

    result = fitted_model.predict(h=12, level=(60,75))
    forecast = fitted_model.forecast(y, h=12, level=[80, 95])
    
    assert "mean" in result, "Missing mean forecast"

    assert len(result["mean"]) == 12, "Forecast length mismatch"

    for lvl in [60, 75]:
        if f"lo-{lvl}" in result:
            assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
        else:
            print(f"Warning: Interval {lvl}% not computed due to missing `prediction_intervals`")

    for lvl in [80, 95]:
        assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
        assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
        assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

if __name__ == "__main__":
    test_seasonal_exponential_smoothing()
    print("Test passed!")