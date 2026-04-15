import jax.numpy as jnp
import pytest
from chronax.utils import ConformalIntervals
from chronax.models import Theta, AutoTheta



# Test Cases
def test_autotheta():
    # simple increasing series
    y = jnp.arange(24.0)

    ci = ConformalIntervals(method="conformal_distribution")
    model = AutoTheta(conformal_params=ci)

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
    model = Theta(conformal_params=ci)

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


def test_autotheta_config_alias_consistency():
    cfg = ConformalIntervals(n_windows=3, h=2, method="conformal_signed")
    model = AutoTheta(conformal_params=cfg)

    assert model.conformal_params is cfg
    assert model.prediction_intervals is cfg

    y = jnp.arange(36.0)
    model.fit(y)
    out = model.predict(h=2, level=[80])
    assert "lo-80" in out and "hi-80" in out


def test_autotheta_prediction_intervals_emits_deprecation_warning():
    cfg = ConformalIntervals(n_windows=3, h=2, method="conformal_signed")
    with pytest.warns(DeprecationWarning, match="prediction_intervals is deprecated"):
        model = AutoTheta(prediction_intervals=cfg)
    assert model.conformal_params is cfg
    assert model.prediction_intervals is cfg


def test_theta_prediction_intervals_emits_deprecation_warning():
    cfg = ConformalIntervals(method="conformal_distribution")
    with pytest.warns(DeprecationWarning, match="prediction_intervals is deprecated"):
        model = Theta(prediction_intervals=cfg)
    assert model.conformal_params is cfg
    assert model.prediction_intervals is cfg


if __name__ == "__main__":
    test_autotheta()
    test_theta()
    test_autotheta_config_alias_consistency()
    test_autotheta_prediction_intervals_emits_deprecation_warning()
    test_theta_prediction_intervals_emits_deprecation_warning()
