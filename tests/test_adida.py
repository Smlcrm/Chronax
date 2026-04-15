import jax.numpy as jnp
import pytest
from chronax.models import ADIDA
from chronax.utils import ConformalIntervals

# Test Cases
def test_adida():
    # simple increasing series
    y = jnp.arange(24.0)

    ci = ConformalIntervals(method="conformal_distribution")
    model = ADIDA(conformal_params=ci)

    fitted_model = model.fit(y)

    result = fitted_model.predict(h=12, level=[60, 75])
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

    print("Test passed!")


def test_adida_rejects_prediction_intervals_constructor_arg():
    cfg = ConformalIntervals(method="conformal_distribution")
    with pytest.raises(TypeError):
        ADIDA(prediction_intervals=cfg)


if __name__ == "__main__":
    test_adida()
