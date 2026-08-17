"""Tests for HINT (chronax.models.hint) — hierarchical bootstrap-reconciliation
wrapper.

Sections: reconciliation matrices -> ctor validation -> fit/predict -> pickle.
P-matrix goldens were generated from neuralforecast 3.1.7's get_bottomup_P /
get_mintrace_ols_P / get_mintrace_wls_P on the two summing matrices below.
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.hint import HINT, get_bottomup_P, get_mintrace_ols_P, get_mintrace_wls_P
from chronax.models.mlp import GMM, MLP

# =============================================================================
# Reconciliation matrices
# =============================================================================

S3 = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
S7 = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

NF_GET_BOTTOMUP_P_S3 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
NF_GET_MINTRACE_OLS_P_S3 = [[0.3333333333333333, 0.6666666666666667, -0.3333333333333333], [0.3333333333333333, -0.3333333333333333, 0.6666666666666667]]
NF_GET_MINTRACE_WLS_P_S3 = [[0.25, 0.75, -0.25], [0.25, -0.25, 0.75]]
NF_GET_BOTTOMUP_P_S7 = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
NF_GET_MINTRACE_OLS_P_S7 = [[0.14285714285714296, 0.23809523809523792, -0.09523809523809537, 0.6190476190476191, -0.3809523809523809, -0.04761904761904759, -0.04761904761904759], [0.14285714285714296, 0.23809523809523792, -0.09523809523809537, -0.3809523809523809, 0.6190476190476191, -0.04761904761904759, -0.04761904761904759], [0.14285714285714296, -0.0952380952380954, 0.23809523809523792, -0.04761904761904756, -0.04761904761904756, 0.6190476190476191, -0.3809523809523809], [0.14285714285714296, -0.0952380952380954, 0.23809523809523792, -0.04761904761904756, -0.04761904761904756, -0.3809523809523809, 0.6190476190476191]]
NF_GET_MINTRACE_WLS_P_S7 = [[0.08333333333333337, 0.2083333333333333, -0.041666666666666706, 0.7083333333333334, -0.29166666666666663, -0.041666666666666664, -0.041666666666666664], [0.08333333333333337, 0.2083333333333333, -0.041666666666666706, -0.29166666666666663, 0.7083333333333334, -0.041666666666666664, -0.041666666666666664], [0.08333333333333334, -0.04166666666666667, 0.20833333333333337, -0.04166666666666667, -0.04166666666666667, 0.7083333333333333, -0.29166666666666674], [0.08333333333333334, -0.04166666666666667, 0.20833333333333337, -0.04166666666666667, -0.04166666666666667, -0.29166666666666674, 0.7083333333333333]]

_P_CASES = [
    (get_bottomup_P, S3, NF_GET_BOTTOMUP_P_S3),
    (get_mintrace_ols_P, S3, NF_GET_MINTRACE_OLS_P_S3),
    (get_mintrace_wls_P, S3, NF_GET_MINTRACE_WLS_P_S3),
    (get_bottomup_P, S7, NF_GET_BOTTOMUP_P_S7),
    (get_mintrace_ols_P, S7, NF_GET_MINTRACE_OLS_P_S7),
    (get_mintrace_wls_P, S7, NF_GET_MINTRACE_WLS_P_S7),
]


class TestReconciliationMatrices:
    @pytest.mark.parametrize("fn,S,golden", _P_CASES,
                             ids=[f"{f.__name__}-{'S3' if s is S3 else 'S7'}" for f, s, _ in _P_CASES])
    def test_matches_nf_golden(self, fn, S, golden):
        np.testing.assert_allclose(fn(S=S), np.asarray(golden), atol=1e-12)

    @pytest.mark.parametrize("fn", [get_bottomup_P, get_mintrace_ols_P, get_mintrace_wls_P])
    @pytest.mark.parametrize("S", [S3, S7], ids=["S3", "S7"])
    def test_PS_is_identity(self, fn, S):
        # P is a right inverse of S for every method: reconciling already-coherent
        # values is a no-op.
        P = fn(S=S)
        np.testing.assert_allclose(P @ S, np.eye(S.shape[1]), atol=1e-10)

    @pytest.mark.parametrize("fn", [get_bottomup_P, get_mintrace_ols_P, get_mintrace_wls_P])
    @pytest.mark.parametrize("S", [S3, S7], ids=["S3", "S7"])
    def test_SP_projects_onto_coherent_subspace(self, fn, S):
        SP = S @ fn(S=S)
        np.testing.assert_allclose(SP @ SP, SP, atol=1e-10)   # idempotent
        rng = np.random.default_rng(0)
        x = rng.normal(size=(S.shape[0], 5))
        rec = SP @ x
        n_agg = S.shape[0] - S.shape[1]
        A = S[:n_agg]
        np.testing.assert_allclose(rec[:n_agg], A @ rec[n_agg:], atol=1e-10)


# =============================================================================
# Ctor validation
# =============================================================================

def _base(h=4, **kw):
    cfg = dict(h=h, input_size=12, hidden_size=8, num_layers=2, max_steps=20,
               windows_batch_size=16, random_seed=0,
               loss=GMM(n_components=2, num_samples=128))
    cfg.update(kw)
    return MLP(**cfg)


class TestHINTCtor:
    def test_h_mismatch_msg(self):
        with pytest.raises(ValueError, match="does not match HINT h"):
            HINT(h=6, S=S3, model=_base(h=4), reconciliation="BottomUp")

    def test_non_distribution_loss_msg(self):
        with pytest.raises(ValueError, match="not a probabilistic objective"):
            HINT(h=4, S=S3, model=_base(loss="mae"), reconciliation="BottomUp")

    def test_unknown_reconciliation_msg(self):
        with pytest.raises(ValueError, match="not available"):
            HINT(h=4, S=S3, model=_base(), reconciliation="TopDown")

    def test_s_without_bottom_identity_raises(self):
        bad = np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])  # bottom rows permuted
        with pytest.raises(ValueError, match="identity"):
            HINT(h=4, S=bad, model=_base(), reconciliation="BottomUp")

    def test_identity_reconciliation_sp_none(self):
        m = HINT(h=4, S=S3, model=_base(), reconciliation="Identity")
        assert m.SP is None

    def test_sp_matches_s_at_p(self):
        m = HINT(h=4, S=S3, model=_base(), reconciliation="MinTraceOLS")
        np.testing.assert_allclose(m.SP, S3 @ np.asarray(NF_GET_MINTRACE_OLS_P_S3), atol=1e-12)

    def test_non_mlp_base_raises(self):
        class Fake:
            h = 4
        with pytest.raises(ValueError, match="MLP"):
            HINT(h=4, S=S3, model=Fake(), reconciliation="BottomUp")


# =============================================================================
# Fit / predict
# =============================================================================

def _hier_y(T=72, seed=0):
    """3-node hierarchy (total = a + b), columns ordered as S3 rows."""
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    a = 10 + 0.2 * t + 2 * np.sin(t / 3.0) + 0.3 * rng.randn(T)
    b = 20 + 0.1 * t + 3 * np.cos(t / 5.0) + 0.3 * rng.randn(T)
    return jnp.asarray(np.stack([a + b, a, b], axis=1), jnp.float32)


def _hint(reconciliation="BottomUp", S=S3, **base_kw):
    return HINT(h=4, S=S, model=_base(**base_kw), reconciliation=reconciliation)


class TestHINTFitPredict:
    def test_hierarchy_shapes(self):
        m = _hint().fit(_hier_y())
        out = m.predict(h=4, level=[80, 95])
        assert set(out) == {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
        for v in out.values():
            assert v.shape == (4, 3)
        assert bool(jnp.all(out["lo-95"] <= out["lo-80"]))
        assert bool(jnp.all(out["hi-80"] <= out["hi-95"]))

    def test_mean_coherence(self):
        m = _hint().fit(_hier_y())
        mean = np.asarray(m.predict(h=4)["mean"])          # [h, 3]
        np.testing.assert_allclose(mean[:, 0], mean[:, 1] + mean[:, 2], rtol=1e-4)

    def test_mean_coherence_mintrace(self):
        m = _hint("MinTraceWLS").fit(_hier_y())
        mean = np.asarray(m.predict(h=4)["mean"])
        np.testing.assert_allclose(mean[:, 0], mean[:, 1] + mean[:, 2], rtol=1e-4)

    def test_sample_coherence(self):
        # Coherence is a per-sample property of the reconciled tensor; marginal
        # quantiles of coherent samples need not (and in general do not) sum.
        from chronax.models.mlp.mlp_training import predict_params
        m = _hint().fit(_hier_y())
        loss = m.model._loss_fn
        args = predict_params(m.model_, m._contexts, input_size=m.model.input_size,
                              scaler=m.model._scaler, loss_fn=loss)
        samples = loss.sample(args, key=jax.random.PRNGKey(m.model.random_seed))
        rec = jnp.einsum("ij,jhs->ihs", jnp.asarray(m.SP, jnp.float32), samples)
        np.testing.assert_allclose(np.asarray(rec[0]), np.asarray(rec[1] + rec[2]),
                                   rtol=1e-4, atol=1e-3)

    def test_identity_is_not_reconciled(self):
        # Identity passes the base samples through: the shared net's raw means are
        # generally incoherent, and BottomUp coherence must change the totals.
        y = _hier_y()
        m_id = _hint("Identity").fit(y)
        m_bu = _hint("BottomUp").fit(y)
        mean_id = np.asarray(m_id.predict(h=4)["mean"])
        mean_bu = np.asarray(m_bu.predict(h=4)["mean"])
        # Bottom rows are untouched by BottomUp; the total row is replaced.
        np.testing.assert_allclose(mean_id[:, 1:], mean_bu[:, 1:], rtol=1e-5)
        assert not np.allclose(mean_id[:, 0], mean_bu[:, 0], rtol=1e-5)

    def test_1d_degenerate_equals_plain_mlp(self):
        # S = [[1]]: pooling over one column is exactly the univariate MLP fit.
        y1 = _hier_y()[:, 1]
        m = HINT(h=4, S=np.eye(1), model=_base(), reconciliation="BottomUp").fit(y1)
        out_h = m.predict(h=4)
        out_m = _base().fit(y1).predict(h=4)
        np.testing.assert_array_equal(np.asarray(out_h["mean"]), np.asarray(out_m["mean"]))
        assert out_h["mean"].shape == (4,)

    def test_rank_follows_input(self):
        m2 = _hint().fit(_hier_y())
        assert m2.predict(h=4)["mean"].ndim == 2
        m1 = HINT(h=4, S=np.eye(1), model=_base(), reconciliation="BottomUp").fit(_hier_y()[:, 1])
        assert m1.predict(h=4)["mean"].ndim == 1

    def test_deterministic_predict(self):
        m = _hint().fit(_hier_y())
        o1 = m.predict(h=4, level=[80])
        o2 = m.predict(h=4, level=[80])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o2[k]))

    def test_model_config_carrier_not_mutated(self):
        m = _hint()
        base = m.model
        m.fit(_hier_y())
        assert base.model_ is None and base._train_y is None

    def test_forecast_stateless(self):
        y = _hier_y()
        o1 = _hint().forecast(y, h=4, level=[80])
        o2 = _hint().fit(y).predict(h=4, level=[80])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o2[k]))

    def test_1d_y_with_wide_s_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            _hint().fit(_hier_y()[:, 0])

    def test_y_width_mismatch_raises(self):
        with pytest.raises(ValueError, match="columns"):
            _hint().fit(_hier_y()[:, :2])

    def test_hist_exog_fit_predict(self):
        # X = shared-calendar historical exog, broadcast to every hierarchy
        # series by the base model's 2-D hist path; predict needs no future X.
        X = jnp.asarray(np.random.RandomState(1).randn(72, 2), jnp.float32)
        m = _hint().fit(_hier_y(), X=X)
        assert m._hist_size == 2
        out = m.predict(h=4)
        assert out["mean"].shape == (4, 3) and bool(jnp.all(jnp.isfinite(out["mean"])))

    def test_hist_exog_forecast_threads_X_and_rejects_futr(self):
        X = jnp.asarray(np.random.RandomState(1).randn(72, 2), jnp.float32)
        assert _hint().forecast(_hier_y(), 4, X=X)["mean"].shape == (4, 3)
        with pytest.raises(ValueError, match="future-known"):
            _hint().forecast(_hier_y(), 4, X_future=X[:4])

    def test_hist_exog_predict_and_conformity_reject_X(self):
        X = jnp.asarray(np.random.RandomState(1).randn(72, 2), jnp.float32)
        m = _hint().fit(_hier_y(), X=X)
        with pytest.raises(ValueError, match="predict takes no X"):
            m.predict(h=4, X=X)
        with pytest.raises(ValueError, match="historical exog"):
            _hint().conformity_scores(_hier_y()[:, 0], X=X)

    def test_hist_exog_pickle_roundtrip(self):
        # The pickle rebuild reads _hist_size off HINT (self.model is the
        # never-mutated config carrier), so the hist-widened net restores cleanly.
        X = jnp.asarray(np.random.RandomState(1).randn(72, 2), jnp.float32)
        m = _hint().fit(_hier_y(), X=X)
        o1 = m.predict(h=4)["mean"]
        o2 = pickle.loads(pickle.dumps(m)).predict(h=4)["mean"]
        np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))

    def test_h_gt_trained_raises(self):
        m = _hint().fit(_hier_y())
        with pytest.raises(ValueError, match="h"):
            m.predict(h=9)

    def test_fitted_raises(self):
        with pytest.raises(NotImplementedError, match="fitted"):
            _hint().forecast(_hier_y(), h=4, fitted=True)

    def test_conformity_scores_2d_raises(self):
        from chronax.utils import ConformalIntervals
        m = _hint()
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_hier_y())
        with pytest.raises(ValueError, match="native"):
            m.conformity_scores(_hier_y())

    def test_conformity_scores_1d_direct_call_leaves_estimator_usable(self):
        # The DIRECT call is HINT's advertised conformal surface (predict is
        # native-only), so the walk-forward must run on an internal clone: the
        # CV vmap re-fits per window, and those fit() writes would otherwise
        # leave escaped tracers on this estimator.
        from chronax.utils import ConformalIntervals
        y1 = _hier_y()[:, 1]
        m = HINT(h=4, S=np.eye(1), model=_base(max_steps=3), reconciliation="BottomUp")
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(y1)
        cs = m.conformity_scores(y1)
        assert cs.shape == (2, 4) and bool(jnp.all(jnp.isfinite(cs)))
        out = m.predict(h=4, level=[80])          # must not raise UnexpectedTracerError
        assert bool(jnp.all(jnp.isfinite(out["mean"])))
        pickle.loads(pickle.dumps(m)).predict(h=4)

    def test_predict_level_twice_then_pickle_then_predict(self):
        m = _hint().fit(_hier_y())
        o1 = m.predict(h=4, level=[90])
        m.predict(h=4, level=[90])
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[90])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o3[k]))
