import jax
import jax.numpy as jnp
from jax import jit
from typing import Optional, List, Dict, Union
from utils import (
    _seasonal_naive,
    _repeat_val_seas,
    ensure_float,
    calculate_sigma,
    _calculate_intervals,
    _quantiles,
    _store_cs,
    _add_fitted_pi,
    _add_conformal_distribution_intervals,
    _get_conformal_method,
)
from conformal_intervals import ConformalIntervals
from base_forecaster import BaseForecaster
from seasonal_naive import SeasonalNaive
   
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

if __name__ == "__main__":
    test_seasonal_naive()
    print("Test passed!")