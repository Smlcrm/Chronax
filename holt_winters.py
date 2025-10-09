#implement everything where everything is a parameter -> 
#input: hyperparams
#see simple exp smoothing
#regular ets
#use pytest

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from conformal_intervals import (
    ConformalIntervals,
)
from auto_ets import AutoETS

class HoltWinters(AutoETS):
    r"""Holt-Winters' method.

    Also known as triple exponential smoothing, Holt-Winters' method is an extension of exponential smoothing for series that contain both trend and seasonality.
    This implementation returns the corresponding `ETS` model with additive (A) or multiplicative (M) errors (so either 'AAA' or 'MAM').

    References:
        - [Rob J. Hyndman and George Athanasopoulos (2018). "Forecasting principles and practice, Methods with seasonality"](https://otexts.com/fpp3/holt-winters.html).

    Args:
        season_length (int): Number of observations per unit of time. Ex: 12 Monthly data.
        error_type (str): The type of error of the ETS model. Can be additive (A) or multiplicative (M).
        alias (str): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals.
            By default, the model will compute the native prediction
            intervals.
    """

    def __init__(
        self,
        season_length: int = 1,  # season length
        error_type: str = "A",  # error type
        alias: str = "HoltWinters",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        self.season_length = season_length
        self.error_type = error_type
        self.alias = alias
        model = error_type + "A" + error_type
        super().__init__(
            season_length, model, alias=alias, prediction_intervals=prediction_intervals
        )


# -----------------
# Helper assertions
# -----------------

def _arr_close(a, b, tol=1e-6):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return bool(jnp.all(jnp.abs(a - b) <= tol))

def _has_keys(d: Dict, keys: List[str]) -> bool:
    return all(k in d for k in keys)


# -------------
# Test cases
# -------------

def test_init_builds_correct_model_string():
    mA = HoltWinters(season_length=4, error_type="A")
    assert mA.model == "AAA", f"expected 'AAA', got {mA.model}"
    mM = HoltWinters(season_length=4, error_type="M")
    assert mM.model == "MAM", f"expected 'MAM', got {mM.model}"
    print("test_init_builds_correct_model_string: OK")

def test_fit_and_basic_predict_no_intervals():
    # length 16 (> npars+4) to satisfy ETS tiny-dataset guard for AAA/MAM
    y = jnp.asarray([10., 12., 13., 11.] * 4)  # season_length=4 pattern -> n=16
    m = HoltWinters(season_length=4, error_type="A")
    m.fit(y)
    out = m.predict(h=3, level=None)
    assert _has_keys(out, ["mean"])
    assert out["mean"].shape == (3,)
    print(out["mean"])
    print("test_fit_and_basic_predict_no_intervals: OK")

def test_predict_with_native_intervals():
    y = jnp.asarray([10., 12., 13., 11.] * 4)  # n=16
    m = HoltWinters(season_length=4, error_type="A")
    m.fit(y)
    out = m.predict(h=5, level=[80, 95])
    # mean + intervals present
    assert _has_keys(out, ["mean", "lo-95", "hi-95", "lo-80", "hi-80"])
    assert out["mean"].shape == (5,)
    for lv in [80, 95]:
        print(out[f"lo-{lv}"])
        print(out[f"mean"], "mean")
    # sanity: lo <= mean <= hi
    for lv in [80, 95]:
        assert jnp.all(out[f"lo-{lv}"] <= out["mean"])
        assert jnp.all(out["mean"] <= out[f"hi-{lv}"])
    print("test_predict_with_native_intervals: OK")

def test_forecast_stateless_matches_shapes_and_optionally_fitted():
    y = jnp.asarray([5., 7., 9., 6.] * 4)  # n=16
    m = HoltWinters(season_length=4, error_type="M")
    # Stateless forecast (no fit)
    out = m.forecast(y=y, h=4, level=None, fitted=False)
    assert _has_keys(out, ["mean"])
    assert out["mean"].shape == (4,)
    # print mean (stateless)
    print(out["mean"], "mean (stateless)")

    # With fitted=True return 'fitted' and native intervals
    out2 = m.forecast(y=y, h=3, level=[90], fitted=True)
    assert _has_keys(out2, ["mean", "fitted", "lo-90", "hi-90"])
    assert out2["mean"].shape == (3,)
    assert out2["fitted"].ndim == 1

    # print fitted + intervals like in test_predict_with_native_intervals
    print(out2["fitted"], "fitted")
    print(out2["lo-90"])
    print(out2["mean"], "mean")
    print(out2["hi-90"])

    # sanity: intervals ordered around mean
    assert jnp.all(out2["lo-90"] <= out2["mean"])
    assert jnp.all(out2["mean"] <= out2["hi-90"])
    print("test_forecast_stateless_matches_shapes_and_optionally_fitted: OK")

def test_forward_uses_existing_fit_and_outputs_intervals():
    y_fit = jnp.asarray([1., 2., 3., 2.] * 4)  # n=16
    y_new = jnp.asarray([2., 3., 4., 3.] * 4)  # n=16
    m = HoltWinters(season_length=4, error_type="A")
    m.fit(y_fit)
    out = m.forward(y=y_new, h=4, level=[80, 95], fitted=False)
    assert _has_keys(out, ["mean", "lo-80", "hi-80", "lo-95", "hi-95"])
    assert out["mean"].shape == (4,)

    # prints, like in test_predict_with_native_intervals
    for lv in [80, 95]:
        print(out[f"lo-{lv}"])
        print(out["mean"], "mean")
        print(out[f"hi-{lv}"])

    # sanity: intervals ordered around mean
    for lv in [80, 95]:
        assert jnp.all(out[f"lo-{lv}"] <= out["mean"])
        assert jnp.all(out["mean"] <= out[f"hi-{lv}"])
    print("test_forward_uses_existing_fit_and_outputs_intervals: OK")


def test_predict_in_sample_option():
    y = jnp.asarray([10., 20., 15., 25.] * 4)  # n=16
    m = HoltWinters(season_length=4, error_type="A")
    m.fit(y)
    ins = m.predict_in_sample(level=None)
    assert _has_keys(ins, ["fitted"])
    assert ins["fitted"].shape == (y.shape[0],)
    # prints
    print(ins["fitted"], "fitted (no intervals)")

    # with intervals over fitted values
    ins2 = m.predict_in_sample(level=[80])
    assert _has_keys(ins2, ["fitted", "fitted-lo-80", "fitted-hi-80"])
    assert ins2["fitted"].shape == (y.shape[0],)
    # prints (like interval tests)
    print(ins2["fitted-lo-80"])
    print(ins2["fitted"], "fitted (with 80)")
    print(ins2["fitted-hi-80"])

    print("test_predict_in_sample_option: OK")


def test_error_type_mapping_only_A_or_M_semantics():
    mA = HoltWinters(season_length=6, error_type="A")
    mM = HoltWinters(season_length=6, error_type="M")
    assert mA.model == "AAA", "HoltWinters 'A' must map to AAA"
    assert mM.model == "MAM", "HoltWinters 'M' must map to MAM"
    print("test_error_type_mapping_only_A_or_M_semantics: OK")


if __name__ == "__main__":
    print("Running HoltWinters tests...")
    test_init_builds_correct_model_string()
    print("____________________________________________")
    test_fit_and_basic_predict_no_intervals()
    print("____________________________________________")
    test_predict_with_native_intervals()
    print("____________________________________________")
    test_forecast_stateless_matches_shapes_and_optionally_fitted()
    print("____________________________________________")
    test_forward_uses_existing_fit_and_outputs_intervals()
    print("____________________________________________")
    test_predict_in_sample_option()
    print("____________________________________________")
    test_error_type_mapping_only_A_or_M_semantics()
    print("____________________________________________")
    print("All HoltWinters tests passed.")