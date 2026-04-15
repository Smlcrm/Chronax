import jax
import jax.numpy as jnp
import pytest
from jax import jit
from typing import Optional, List, Dict, Union
from chronax.utils import (
    _seasonal_naive,
    _repeat_val_seas,
    ensure_float,
    calculate_sigma,
    _calculate_intervals,
    _quantiles,
    _add_fitted_pi,
)
from chronax.utils import ConformalIntervals
from chronax.models.base_forecaster import BaseForecaster
from chronax.models import SeasonalNaive
   
# Test Cases
def test_seasonal_naive():
    y = jnp.arange(24.0)

    model = SeasonalNaive(season_length=12)
    fitted_model = model.fit(y)

    result = fitted_model.predict(h=12, level=(60,75))
    forecast = fitted_model.forecast(y, h=12, level=[80, 95])
    
    assert "mean" in result, "Missing mean forecast"

    assert len(result["mean"]) == 12, "Forecast length mismatch"

    for lvl in [60, 75]:
        assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"
        assert f"lo-{lvl}" in result, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in result, f"Missing upper bound for {lvl}% interval"

    for lvl in [80, 95]:
        assert f"lo-{lvl}" in forecast, f"Missing lower bound for {lvl}% interval"
        assert f"hi-{lvl}" in forecast, f"Missing upper bound for {lvl}% interval"
        assert len(forecast[f"lo-{lvl}"]) == 12, f"Lower interval {lvl}% has wrong length"
        assert len(forecast[f"hi-{lvl}"]) == 12, f"Upper interval {lvl}% has wrong length"


def test_seasonal_naive_conformal_params_path():
    y = jnp.arange(48.0)
    cfg = ConformalIntervals(n_windows=3, h=2, method="conformal_distribution")
    model = SeasonalNaive(season_length=12, conformal_params=cfg).fit(y)

    assert model.conformal_params is cfg

    res = model.predict(h=2, level=[80])
    assert "lo-80" in res and "hi-80" in res


if __name__ == "__main__":
    test_seasonal_naive()
    test_seasonal_naive_conformal_params_path()
    print("Test passed!")