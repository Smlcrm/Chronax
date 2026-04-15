"""Tests for ``chronax.utils.conformal_workflow`` orchestration helpers."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from chronax.utils.conformal_intervals import ConformalIntervals
from chronax.utils.conformal_workflow import (
    add_conformal_intervals,
    compute_conformity_scores,
    resolve_conformal_params,
)


class _DummyForecaster:
    """Minimal stand-in with conformal config and vmap-safe ``conformity_scores``."""

    def __init__(self):
        self.conformal_params = ConformalIntervals(n_windows=2, h=2, method="conformal_distribution")
        self.prediction_intervals = None
        self._scores_calls = 0

    def conformity_scores(self, y, X):
        self._scores_calls += 1
        del X  # unused in dummy
        del y
        return jnp.array(
            [[0.5, -0.25], [-0.1, 0.3]],
            dtype=jnp.float32,
        )


def test_resolve_conformal_params_returns_config():
    model = _DummyForecaster()
    cfg = resolve_conformal_params(model)
    assert cfg is not None
    assert cfg.method == "conformal_distribution"
    assert cfg.n_windows == 2
    assert cfg.h == 2
    assert model.conformal_params is cfg
    assert model.prediction_intervals is cfg


def test_add_conformal_intervals_uses_model_scores_and_adds_lo_hi_keys():
    model = _DummyForecaster()
    fcst = {"mean": jnp.array([1.0, 2.0], dtype=jnp.float32)}
    y = jnp.arange(8.0, dtype=jnp.float32)

    out = add_conformal_intervals(model=model, fcst=fcst, y=y, X=None, level=[80])

    assert model._scores_calls == 1
    assert "lo-80" in out
    assert "hi-80" in out
    assert out["lo-80"].shape == (2,)
    assert out["hi-80"].shape == (2,)


def test_compute_conformity_scores_shape():
    model = _DummyForecaster()
    y = jnp.arange(10.0, dtype=jnp.float32)
    cs = compute_conformity_scores(model, y=y, X=None)
    assert cs.ndim == 2
    assert cs.shape == (2, 2)


def test_resolve_conformal_params_prefers_conformal_params_when_both_set():
    a = ConformalIntervals(n_windows=3, h=1, method="conformal_signed")
    b = ConformalIntervals(n_windows=2, h=2, method="conformal_distribution")
    model = _DummyForecaster()
    model.conformal_params = a
    model.prediction_intervals = b
    cfg = resolve_conformal_params(model)
    assert cfg is a
    assert cfg.method == "conformal_signed"


def test_add_conformal_intervals_requires_stored_scores_when_y_is_none():
    model = _DummyForecaster()
    fcst = {"mean": jnp.ones(2, dtype=jnp.float32)}
    with pytest.raises(ValueError, match="Conformity scores are missing"):
        add_conformal_intervals(model=model, fcst=fcst, y=None, X=None, level=[80])


def test_migration_doc_exists_and_mentions_breaking_change():
    doc_path = Path("docs/migration/conformal-params.md")
    assert doc_path.exists()
    text = doc_path.read_text(encoding="utf-8")
    assert "prediction_intervals" in text
    assert "has been removed" in text
