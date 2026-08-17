
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


import numpy as np
from chronax.models.theta import theta_model as _tm


def test_autotheta_forward_dispatches_on_winning_variant():
    # When the auto fit selects a non-STM variant, forward() must re-fit with
    # THAT variant (dispatch on model_type_int), not the stale fits[0] "STM"
    # string left by the merge.
    rng = np.random.default_rng(0)
    n = 120
    t = np.arange(n)
    y = jnp.asarray(10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n),
                    dtype=jnp.float64)
    m = AutoTheta(season_length=12)
    m.fit(y)
    mt_int = int(m.model_["model_type_int"])
    # forward with the same series must reproduce the fitted forecast closely
    # (same variant), which the STM-forced dispatch would not for OTM/DOTM.
    fc_fit = np.asarray(m.predict(h=6)["mean"])
    fc_fwd = np.asarray(m.forward(y=y, h=6)["mean"])
    if mt_int != 1:  # a non-STM winner: STM dispatch would diverge materially
        np.testing.assert_allclose(fc_fwd, fc_fit, rtol=0.05)


def test_autotheta_mc_intervals_finite_and_ordered():
    rng = np.random.default_rng(1)
    n = 120
    t = np.arange(n)
    y = jnp.asarray(20 + 0.3 * t + 4 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n),
                    dtype=jnp.float64)
    m = AutoTheta(season_length=12)
    m.fit(y)
    out = m.predict(h=8, level=[80, 95])
    for k in ("lo-80", "hi-80", "lo-95", "hi-95"):
        assert np.all(np.isfinite(np.asarray(out[k])))
    assert np.all(np.asarray(out["lo-95"]) <= np.asarray(out["lo-80"]))
    assert np.all(np.asarray(out["hi-80"]) <= np.asarray(out["hi-95"]))


def test_theta_model_compat_shim_resolves():
    assert _tm.AutoTheta is AutoTheta
    assert _tm.Theta is Theta


if __name__ == "__main__":
    test_autotheta()
    test_theta()
