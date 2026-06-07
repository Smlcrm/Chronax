"""Tests for chronax.models.patchtst.patchtst_model — BaseForecaster conformance."""
import pickle

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.patchtst.patchtst_model import PatchTST


def _make_y(n=200):
    return jnp.asarray(np.sin(np.arange(n) / 10.0), dtype=jnp.float32)


def _tiny():
    return PatchTST(h=12, input_size=36, hidden_size=16, n_heads=2, encoder_layers=1,
                    linear_hidden_size=32, max_steps=20, windows_batch_size=64, random_seed=0)


def test_patchtst_inherits_baseforecaster():
    assert issubclass(PatchTST, BaseForecaster)


def test_init_sets_conformal_params_to_none():
    assert _tiny().conformal_params is None


def test_fit_returns_self_and_sets_model():
    m = _tiny()
    out = m.fit(_make_y())
    assert out is m
    assert m.model_ is not None


def test_predict_deterministic_with_same_seed():
    y = _make_y()
    p1 = _tiny().fit(y).predict(h=12)["mean"]
    p2 = _tiny().fit(y).predict(h=12)["mean"]
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), rtol=1e-5, atol=1e-5)


def test_model_beats_naive_last_value_on_easy_signal():
    # End-to-end correctness floor: catches broken loss/optimizer/RevIN-denorm/
    # flatten-order wiring that still passes shape + determinism tests.
    n = 200
    y = jnp.asarray(np.sin(np.arange(n) / 5.0), dtype=jnp.float32)
    m = PatchTST(h=12, input_size=36, hidden_size=32, n_heads=4, encoder_layers=2,
                 linear_hidden_size=64, max_steps=200, windows_batch_size=64, random_seed=0)
    m.fit(y[:-12])
    pred = np.asarray(m.predict(h=12)["mean"])
    y_true = np.asarray(y[-12:])
    naive = np.full(12, float(y[-13]))
    assert np.mean(np.abs(pred - y_true)) < np.mean(np.abs(naive - y_true))


def test_predict_default_returns_h_steps():
    m = _tiny().fit(_make_y())
    assert m.predict(h=12)["mean"].shape == (12,)


def test_predict_smaller_h_slices():
    m = _tiny().fit(_make_y())
    assert m.predict(h=5)["mean"].shape == (5,)


def test_predict_larger_h_raises():
    m = _tiny().fit(_make_y())
    with pytest.raises(ValueError, match="not supported"):
        m.predict(h=13)


def test_fit_raises_on_exog():
    with pytest.raises(NotImplementedError):
        _tiny().fit(_make_y(), X=jnp.ones((200, 1)))


def test_fit_raises_on_short_series():
    with pytest.raises(ValueError, match="too short"):
        _tiny().fit(_make_y(40))  # < input_size + h = 48


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        _tiny().predict(h=12)


def test_hidden_not_divisible_by_heads_raises_at_fit():
    m = PatchTST(h=12, input_size=36, hidden_size=30, n_heads=4, encoder_layers=1,
                 max_steps=2, windows_batch_size=8, random_seed=0)
    with pytest.raises(ValueError, match="divisible"):
        m.fit(_make_y())
