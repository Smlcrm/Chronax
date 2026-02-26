
import jax.numpy as jnp
from chronax.utils import ConformalIntervals
from chronax.models import Theta, AutoTheta



# Test Cases
def test_autotheta():
    # simple increasing series
    y = jnp.arange(24.0)

    ci = ConformalIntervals(method="conformal_distribution")
    model = AutoTheta(prediction_intervals=ci)

    # --- Fit the model ---
    fitted_model = model.fit(y)

    # --- Predict using fitted model ---
    result = fitted_model.predict(h=12, level=[60, 75])

    # --- Forecast directly from raw series ---
    forecast = fitted_model.forecast(y, h=12, level=[80, 95])

    # ---- Assertions ----
    assert "mean" in result, "Missing mean forecast"
    assert len(result["mean"]) == 12, "Forecast length mismatch"

    for lvl in [60, 75]:
        assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
        assert len(result[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
        assert len(result[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

    for lvl in [80, 95]:
        assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
        assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
        assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"

    print("AutoTheta test passed!")


def test_theta():
    # Basic variant using the fixed STM model
    y = jnp.arange(24.0)

    ci = ConformalIntervals(method="conformal_distribution")
    model = Theta(prediction_intervals=ci)

    fitted_model = model.fit(y)
    result = fitted_model.predict(h=6, level=[70, 90])
    forecast = fitted_model.forecast(y, h=6, level=[90, 95])

    assert "mean" in result, "Missing mean forecast"
    assert len(result["mean"]) == 6, "Forecast length mismatch"

    for lvl in [70, 90]:
        assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"

    for lvl in [90, 95]:
        assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"

    print("Theta test passed!")


if __name__ == "__main__":
    test_autotheta()
    test_theta()
