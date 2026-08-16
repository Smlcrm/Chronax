"""Tests for the MLP forecaster package (chronax.models.mlp).

Sections: Losses (GMM distribution loss) -> Module -> Training -> Model.
GMM golden values were generated from neuralforecast 3.1.7's GMM on fixed
float64 tensors (see the anchored/bare scale_decouple and NLL constants).
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models.mlp.mlp_losses import GMM, weighted_average

# =============================================================================
# Losses — GMM distribution loss
# =============================================================================

# ---- neuralforecast 3.1.7 GMM goldens (float64) ----
NF_MEANS_U = [[[-1.0, -0.8235294117647058, -0.6470588235294117], [-0.47058823529411764, -0.2941176470588235, -0.11764705882352933], [0.05882352941176472, 0.23529411764705888, 0.41176470588235303]], [[0.5882352941176472, 0.7647058823529413, 0.9411764705882355], [1.1176470588235294, 1.2941176470588238, 1.4705882352941178], [1.6470588235294121, 1.823529411764706, 2.0]]]
NF_STDS_U = [[[0.3, 0.4, 0.5], [0.6000000000000001, 0.7, 0.8], [0.9000000000000001, 1.0, 1.1]], [[1.2, 1.3, 1.4000000000000001], [1.5000000000000002, 1.6, 1.7000000000000002], [1.8, 1.9000000000000001, 2.0]]]
NF_Y = [[0.5, 0.7, 0.9], [1.1, 1.3, 1.5]]
NF_NLL_UNIFORM_K3 = 1.8073118231740637
NF_MEANS_W = [[[0.0, 0.09090909090909091], [0.18181818181818182, 0.2727272727272727], [0.36363636363636365, 0.4545454545454546]], [[0.5454545454545454, 0.6363636363636364], [0.7272727272727273, 0.8181818181818182], [0.9090909090909092, 1.0]]]
NF_STDS_W = [[[0.25, 0.3], [0.35, 0.4], [0.45, 0.5]], [[0.55, 0.6000000000000001], [0.65, 0.7], [0.75, 0.8]]]
NF_WEIGHTS_W = [[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]], [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]]
NF_NLL_WEIGHTED_K2 = 0.7774574534061275
NF_MEANS_C = [[[-0.5, -0.40909090909090906], [-0.3181818181818182, -0.2272727272727273], [-0.13636363636363635, -0.045454545454545414]], [[0.045454545454545414, 0.13636363636363635], [0.2272727272727273, 0.31818181818181823], [0.40909090909090917, 0.5]]]
NF_STDS_C = [[[0.4, 0.42000000000000004], [0.44, 0.46], [0.48000000000000004, 0.5]], [[0.52, 0.54], [0.56, 0.5800000000000001], [0.6000000000000001, 0.62]]]
NF_NLL_BATCH_CORR = 4.381887988574335
NF_NLL_HORIZON_CORR = 6.489795446619692
NF_MASK = [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
NF_NLL_MASKED = 2.0718605839397597
NF_NLL_MASKED_POISON_1E6 = 2.0718605839397597
NF_SD_RAW_MEANS = [[[-2.0, -1.6363636363636362], [-1.2727272727272727, -0.9090909090909092], [-0.5454545454545454, -0.18181818181818166]], [[0.18181818181818166, 0.5454545454545454], [0.9090909090909092, 1.272727272727273], [1.6363636363636367, 2.0]]]
NF_SD_RAW_STDS = [[[-1.0, -0.6363636363636364], [-0.2727272727272727, 0.09090909090909083], [0.4545454545454546, 0.8181818181818183]], [[1.1818181818181817, 1.5454545454545454], [1.9090909090909092, 2.272727272727273], [2.6363636363636367, 3.0]]]
NF_SD_LOC = [[[10.0]], [[-5.0]]]
NF_SD_SCALE = [[[2.0]], [[0.5]]]
NF_SD_MEANS_ANCHORED = [[[6.0, 6.7272727272727275], [7.454545454545455, 8.181818181818182], [8.90909090909091, 9.636363636363637]], [[-4.909090909090909, -4.7272727272727275], [-4.545454545454545, -4.363636363636363], [-4.181818181818182, -4.0]]]
NF_SD_STDS_ANCHORED = [[[1.0265233750364458, 1.2495068945971282], [1.5321047846161049, 1.8792688566508615], [2.292054068482711, 2.767361863594267]], [[0.8246693945726277, 0.9693641756202369], [1.1236497647287982, 1.285394469795624], [1.4527681566590764, 1.6242936757868711]]]
NF_SD_MEANS_BARE = NF_SD_RAW_MEANS
NF_SD_STDS_BARE = [[[0.31326168751822286, 0.4247534472985641], [0.5660523923080525, 0.7396344283254308], [0.9460270342413554, 1.1836809317971335]], [[1.4493387891452554, 1.7387283512404739], [2.047299529457596, 2.370788939591248], [2.7055363133181527, 3.048587351573742]]]
NF_QUANTILES_LEVEL_80_90 = [0.5, 0.05, 0.1, 0.9, 0.95]
NF_QUANTILES_EXPLICIT = [0.1, 0.5, 0.9]


def _f64(x):
    return jnp.asarray(np.asarray(x, dtype=np.float64))


class TestWeightedAverage:
    def test_no_weights_is_plain_mean(self):
        x = _f64([[1.0, 2.0], [3.0, 4.0]])
        assert float(weighted_average(x)) == pytest.approx(2.5)

    def test_denominator_clamped_to_one(self):
        # Weight sum 0.5 < 1 -> denominator is clamped to 1.0 (not eps).
        x = _f64([[2.0, 4.0]])
        w = _f64([[0.25, 0.25]])
        assert float(weighted_average(x, w)) == pytest.approx(2.0 * 0.25 + 4.0 * 0.25)

    def test_zero_weight_masks_nan(self):
        x = _f64([[1.0, jnp.nan]])
        w = _f64([[1.0, 0.0]])
        assert float(weighted_average(x, w)) == pytest.approx(1.0)


class TestGMMConfig:
    def test_outputsize_multiplier(self):
        assert GMM(n_components=1).outputsize_multiplier == 2
        assert GMM(n_components=4).outputsize_multiplier == 8
        assert GMM(n_components=10, weighted=True).outputsize_multiplier == 30

    def test_is_distribution_output(self):
        assert GMM().is_distribution_output is True

    def test_quantiles_from_level(self):
        assert list(GMM(level=[80, 90]).quantiles) == pytest.approx(NF_QUANTILES_LEVEL_80_90)

    def test_quantiles_explicit(self):
        assert list(GMM(quantiles=[0.1, 0.5, 0.9]).quantiles) == pytest.approx(NF_QUANTILES_EXPLICIT)

    def test_return_params_unsupported(self):
        with pytest.raises(NotImplementedError):
            GMM(return_params=True)


class TestGMMDomainMap:
    def test_split_unweighted(self):
        g = GMM(n_components=2)
        raw = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
        means, stds = g.domain_map(raw)
        assert means.shape == (2, 3, 2) and stds.shape == (2, 3, 2)
        np.testing.assert_array_equal(np.asarray(means), np.asarray(raw[..., :2]))
        np.testing.assert_array_equal(np.asarray(stds), np.asarray(raw[..., 2:]))

    def test_split_weighted(self):
        g = GMM(n_components=2, weighted=True)
        raw = jnp.arange(1 * 2 * 6, dtype=jnp.float32).reshape(1, 2, 6)
        means, stds, w = g.domain_map(raw)
        assert means.shape == stds.shape == w.shape == (1, 2, 2)
        np.testing.assert_array_equal(np.asarray(w), np.asarray(raw[..., 4:]))


class TestGMMScaleDecouple:
    def test_bare_softplus_no_floor(self):
        g = GMM(n_components=2)
        means, stds = g.scale_decouple((_f64(NF_SD_RAW_MEANS), _f64(NF_SD_RAW_STDS)))
        np.testing.assert_allclose(np.asarray(means), NF_SD_MEANS_BARE, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(stds), NF_SD_STDS_BARE, rtol=1e-12)

    def test_anchored_affine_and_floor(self):
        g = GMM(n_components=2)
        means, stds = g.scale_decouple(
            (_f64(NF_SD_RAW_MEANS), _f64(NF_SD_RAW_STDS)),
            loc=_f64(NF_SD_LOC), scale=_f64(NF_SD_SCALE),
        )
        np.testing.assert_allclose(np.asarray(means), NF_SD_MEANS_ANCHORED, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(stds), NF_SD_STDS_ANCHORED, rtol=1e-12)

    def test_identity_anchor_still_applies_floor(self):
        # loc=0/scale=1 (the identity scaler's REAL tensors) keeps the +0.2 floor active.
        g = GMM(n_components=2)
        raw_m, raw_s = _f64(NF_SD_RAW_MEANS), _f64(NF_SD_RAW_STDS)
        zeros = jnp.zeros((2, 1, 1), raw_m.dtype)
        ones = jnp.ones((2, 1, 1), raw_m.dtype)
        means, stds = g.scale_decouple((raw_m, raw_s), loc=zeros, scale=ones)
        np.testing.assert_allclose(np.asarray(means), NF_SD_MEANS_BARE, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(stds), np.asarray(_f64(NF_SD_STDS_BARE)) + 0.2, rtol=1e-12)


class TestGMMNll:
    def test_uniform_matches_nf(self):
        g = GMM(n_components=3)
        nll = g(_f64(NF_Y), (_f64(NF_MEANS_U), _f64(NF_STDS_U)))
        assert float(nll) == pytest.approx(NF_NLL_UNIFORM_K3, rel=1e-12)

    def test_weighted_matches_nf(self):
        g = GMM(n_components=2, weighted=True)
        nll = g(_f64(NF_Y), (_f64(NF_MEANS_W), _f64(NF_STDS_W), _f64(NF_WEIGHTS_W)))
        assert float(nll) == pytest.approx(NF_NLL_WEIGHTED_K2, rel=1e-12)

    def test_batch_correlation_matches_nf(self):
        g = GMM(n_components=2, batch_correlation=True)
        nll = g(_f64(NF_Y), (_f64(NF_MEANS_C), _f64(NF_STDS_C)))
        assert float(nll) == pytest.approx(NF_NLL_BATCH_CORR, rel=1e-12)

    def test_horizon_correlation_matches_nf(self):
        g = GMM(n_components=2, horizon_correlation=True)
        nll = g(_f64(NF_Y), (_f64(NF_MEANS_C), _f64(NF_STDS_C)))
        assert float(nll) == pytest.approx(NF_NLL_HORIZON_CORR, rel=1e-12)

    def test_masked_matches_nf(self):
        g = GMM(n_components=3)
        nll = g(_f64(NF_Y), (_f64(NF_MEANS_U), _f64(NF_STDS_U)), mask=_f64(NF_MASK))
        assert float(nll) == pytest.approx(NF_NLL_MASKED, rel=1e-12)

    def test_masked_poison_finite_matches_nf(self):
        g = GMM(n_components=3)
        y = np.asarray(NF_Y, dtype=np.float64)
        y[0, 2], y[1, 1] = 1.0e6, -1.0e6
        nll = g(_f64(y), (_f64(NF_MEANS_U), _f64(NF_STDS_U)), mask=_f64(NF_MASK))
        assert float(nll) == pytest.approx(NF_NLL_MASKED_POISON_1E6, rel=1e-12)

    def test_masked_poison_nan_is_ignored(self):
        # torch's validators reject NaN targets outright; the chronax port must
        # instead neutralize them wherever the mask is zero (zero-weight cells
        # never contribute, even as NaN*0).
        g = GMM(n_components=3)
        y = np.asarray(NF_Y, dtype=np.float64)
        y[0, 2], y[1, 1] = np.nan, np.nan
        nll = g(_f64(y), (_f64(NF_MEANS_U), _f64(NF_STDS_U)), mask=_f64(NF_MASK))
        assert float(nll) == pytest.approx(NF_NLL_MASKED, rel=1e-12)


class TestGMMMeanAndSample:
    def test_analytic_mean_uniform(self):
        g = GMM(n_components=3)
        mean = g.analytic_mean((_f64(NF_MEANS_U), _f64(NF_STDS_U)))
        np.testing.assert_allclose(
            np.asarray(mean), np.mean(np.asarray(NF_MEANS_U), axis=-1), rtol=1e-12
        )

    def test_analytic_mean_weighted(self):
        g = GMM(n_components=2, weighted=True)
        mean = g.analytic_mean((_f64(NF_MEANS_W), _f64(NF_STDS_W), _f64(NF_WEIGHTS_W)))
        expected = np.sum(np.asarray(NF_MEANS_W) * np.asarray(NF_WEIGHTS_W), axis=-1)
        np.testing.assert_allclose(np.asarray(mean), expected, rtol=1e-12)

    def test_sample_shape_and_determinism(self):
        g = GMM(n_components=2, num_samples=64)
        args = (_f64(NF_MEANS_W), _f64(NF_STDS_W))
        k = jax.random.PRNGKey(7)
        s1 = g.sample(args, key=k)
        s2 = g.sample(args, key=k)
        s3 = g.sample(args, key=jax.random.PRNGKey(8))
        assert s1.shape == (2, 3, 64)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
        assert not np.allclose(np.asarray(s1), np.asarray(s3))

    def test_sample_moments_single_component(self):
        g = GMM(n_components=1, num_samples=20000)
        mu, sigma = 2.0, 0.5
        args = (jnp.full((1, 1, 1), mu), jnp.full((1, 1, 1), sigma))
        s = g.sample(args, key=jax.random.PRNGKey(0))
        assert float(jnp.mean(s)) == pytest.approx(mu, abs=0.02)
        assert float(jnp.std(s)) == pytest.approx(sigma, abs=0.02)

    def test_sample_quantiles_monotone(self):
        g = GMM(n_components=2, num_samples=512)
        s = g.sample((_f64(NF_MEANS_W), _f64(NF_STDS_W)), key=jax.random.PRNGKey(1))
        q = jnp.quantile(s, jnp.asarray([0.05, 0.5, 0.95]), axis=-1)
        assert bool(jnp.all(q[0] <= q[1])) and bool(jnp.all(q[1] <= q[2]))

    def test_sampling_matches_grid_resample_distribution(self):
        # The torch pipeline materializes an evenly-spaced quantile grid of the
        # sampled distribution and bootstrap-resamples it; direct MC draws are the
        # same distribution without the grid detour. Compare empirical quantiles
        # of both routes on a bimodal 2-component GMM.
        mu = np.array([-1.0, 2.0])
        sd = np.array([0.5, 0.3])
        n = 8000
        g = GMM(n_components=2, num_samples=n)
        args = (jnp.asarray(mu.reshape(1, 1, 2)), jnp.asarray(sd.reshape(1, 1, 2)))
        ours = np.asarray(g.sample(args, key=jax.random.PRNGKey(3))).ravel()

        rng = np.random.default_rng(0)
        comp = rng.integers(0, 2, size=n)
        base = rng.normal(mu[comp], sd[comp])
        grid = np.quantile(base, np.arange(n) / n)
        resampled = grid[rng.integers(0, n, size=n)]

        # Compare empirical CDFs at fixed probes: quantiles are ill-defined in the
        # near-zero-density valley between the modes, CDF values are not
        # (SE <= sqrt(0.25/n) ~ 0.006 here).
        probes = np.linspace(-2.5, 3.0, 12)
        cdf_ours = (ours[:, None] <= probes).mean(axis=0)
        cdf_ref = (resampled[:, None] <= probes).mean(axis=0)
        np.testing.assert_allclose(cdf_ours, cdf_ref, atol=0.03)


# =============================================================================
# Module — MLPNet
# =============================================================================
from flax import nnx  # noqa: E402

from chronax.models.mlp.mlp_module import MLPNet  # noqa: E402


def _param_count(net) -> int:
    state = nnx.state(net, nnx.Param)
    return sum(int(np.prod(v.shape)) for v in jax.tree.leaves(state))


class TestMLPNet:
    def test_layer_structure_matches_nf(self):
        # NF MLP: num_layers ReLU'd Linears (first: in->hidden, rest hidden->hidden)
        # plus a SEPARATE raw out head Linear(hidden, h*mult).
        net = MLPNet(h=4, input_size=12, num_layers=2, hidden_size=16,
                     outputsize_multiplier=2, rngs=nnx.Rngs(0))
        assert len(net.mlp) == 2
        assert net.mlp[0].kernel.shape == (12, 16)
        assert net.mlp[1].kernel.shape == (16, 16)
        assert net.out.kernel.shape == (16, 8)
        assert _param_count(net) == (12 * 16 + 16) + (16 * 16 + 16) + (16 * 8 + 8)

    def test_first_layer_width_includes_futr(self):
        net = MLPNet(h=4, input_size=12, futr_exog_size=3, num_layers=2,
                     hidden_size=8, outputsize_multiplier=1, rngs=nnx.Rngs(0))
        assert net.mlp[0].kernel.shape[0] == 12 + 3 * (12 + 4)

    def test_forward_shape(self):
        net = MLPNet(h=4, input_size=12, num_layers=2, hidden_size=8,
                     outputsize_multiplier=3, rngs=nnx.Rngs(0))
        out = net(jnp.ones((5, 12, 1)))
        assert out.shape == (5, 4, 3)

    def test_relu_after_every_hidden_layer_and_raw_out(self):
        # Zero kernels + negative biases in the hidden stack: ReLU kills every
        # hidden activation, so the output must be exactly the out head's bias.
        net = MLPNet(h=2, input_size=6, num_layers=2, hidden_size=4,
                     outputsize_multiplier=1, rngs=nnx.Rngs(0))
        for layer in net.mlp:
            layer.kernel.value = jnp.zeros_like(layer.kernel.value)
            layer.bias.value = -jnp.ones_like(layer.bias.value)
        expected = np.asarray(net.out.bias.value).reshape(2, 1)
        out = net(jnp.ones((1, 6, 1)))
        np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-6)


# =============================================================================
# Training
# =============================================================================
from chronax.models.mlp.mlp_scaler import IdentityScaler, RobustScaler  # noqa: E402
from chronax.models.mlp.mlp_losses import mae  # noqa: E402
from chronax.models.mlp.mlp_training import (  # noqa: E402
    build_windows, forward_loss, predict_params, predict_step, train, train_on_windows,
)


def _make_y(T=60):
    t = np.arange(T)
    return jnp.asarray(10 + 0.3 * t + 2 * np.sin(t / 3.0), jnp.float32)


class TestMLPTraining:
    def test_build_windows_padding_semantics(self):
        y = jnp.arange(1.0, 11.0)
        w, m = build_windows(y, input_size=3, h=2)
        assert w.shape == (7, 5) and m.shape == (7, 2)
        np.testing.assert_array_equal(np.asarray(w[0]), [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(np.asarray(w[6]), [7, 8, 9, 10, 0])
        np.testing.assert_array_equal(np.asarray(m[6]), [1, 0])

    def test_distribution_loss_identity_scaler_closed_form(self):
        # A zeroed net emits raw params (0, 0); under the identity scaler's REAL
        # 0/1 anchors the decoupled distribution is N(0, softplus(0)+0.2) in
        # ORIGINAL units, so the masked NLL has a closed form on original y.
        from chronax.models.mlp.mlp_losses import GMM
        L, h = 4, 2
        y = jnp.asarray([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], jnp.float32)  # T=6 -> n=2
        w, m = build_windows(y, L, h)
        net = MLPNet(h=h, input_size=L, num_layers=1, hidden_size=3,
                     outputsize_multiplier=2, rngs=nnx.Rngs(0))
        for layer in list(net.mlp) + [net.out]:
            layer.kernel.value = jnp.zeros_like(layer.kernel.value)
            layer.bias.value = jnp.zeros_like(layer.bias.value)
        loss = forward_loss(net, w, m, h=h, input_size=L, scaler=IdentityScaler(),
                            loss_fn=GMM(n_components=1))
        sigma = float(jax.nn.softplus(0.0)) + 0.2
        targets = np.array([[9.0, 10.0], [10.0, 0.0]])
        mask = np.array([[1.0, 1.0], [1.0, 0.0]])
        nll = 0.5 * (targets / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)
        expected = float((nll * mask).sum() / mask.sum())
        assert float(loss) == pytest.approx(expected, rel=1e-5)

    def test_distribution_loss_robust_scaler_closed_form(self):
        # Same zeroed net under the robust scaler: means collapse to the window
        # median, stds to (softplus(0)+0.2)*scale — verifying the decoupling uses
        # the PER-WINDOW insample stats and the original-scale target.
        from chronax.models.mlp.mlp_losses import GMM
        from chronax.models.mlp.mlp_scaler import RobustScaler as RS
        L, h = 4, 1
        y = jnp.asarray([2.0, 4.0, 6.0, 8.0, 100.0], jnp.float32)  # T=5 -> n=1
        w, m = build_windows(y, L, h)
        net = MLPNet(h=h, input_size=L, num_layers=1, hidden_size=3,
                     outputsize_multiplier=2, rngs=nnx.Rngs(0))
        for layer in list(net.mlp) + [net.out]:
            layer.kernel.value = jnp.zeros_like(layer.kernel.value)
            layer.bias.value = jnp.zeros_like(layer.bias.value)
        loss = forward_loss(net, w, m, h=h, input_size=L, scaler=RS(),
                            loss_fn=GMM(n_components=1))
        scaler = RS()
        shift, scale = scaler.stats(w[:, :L], axis=1)
        mu = float(shift[0, 0])
        sigma = (float(jax.nn.softplus(0.0)) + 0.2) * float(scale[0, 0])
        target = 100.0
        expected = 0.5 * ((target - mu) / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)
        assert float(loss) == pytest.approx(expected, rel=1e-5)

    def test_train_is_vmap_traceable(self):
        from chronax.models.mlp.mlp_losses import GMM
        def run(seed):
            net = MLPNet(h=4, input_size=12, num_layers=2, hidden_size=8,
                         outputsize_multiplier=2, rngs=nnx.Rngs(0))
            return train(net, _make_y(), h=4, input_size=12, max_steps=3,
                         windows_batch_size=16, lr=1e-3, seed=seed,
                         loss_fn=GMM(n_components=1), scaler=IdentityScaler())
        out = jax.vmap(run)(jnp.arange(2))
        assert out.shape == (2, 3) and bool(jnp.all(jnp.isfinite(out)))

    def test_train_runs_and_updates_params(self):
        net = MLPNet(h=4, input_size=12, num_layers=2, hidden_size=8,
                     outputsize_multiplier=1, rngs=nnx.Rngs(0))
        before = np.asarray(net.out.kernel.value).copy()
        losses = train(net, _make_y(), h=4, input_size=12, max_steps=10,
                       windows_batch_size=16, lr=1e-3, seed=0, loss_fn=mae,
                       scaler=RobustScaler())
        assert bool(jnp.all(jnp.isfinite(losses)))
        assert not np.allclose(before, np.asarray(net.out.kernel.value))

    def test_predict_step_batched_matches_single(self):
        net = MLPNet(h=4, input_size=12, num_layers=2, hidden_size=8,
                     outputsize_multiplier=1, rngs=nnx.Rngs(0))
        y = _make_y()
        ctx = jnp.stack([y[-12:], y[-13:-1]])
        single = predict_step(net, y[-12:], h=4, input_size=12, scaler=RobustScaler())
        batched = predict_step(net, ctx, h=4, input_size=12, scaler=RobustScaler())
        assert batched.shape == (2, 4, 1)
        np.testing.assert_array_equal(np.asarray(batched[0]), np.asarray(single))

    def test_predict_params_batched_matches_single(self):
        from chronax.models.mlp.mlp_losses import GMM
        g = GMM(n_components=2)
        net = MLPNet(h=4, input_size=12, num_layers=2, hidden_size=8,
                     outputsize_multiplier=g.outputsize_multiplier, rngs=nnx.Rngs(0))
        y = _make_y()
        ctx1 = y[-12:]
        ctx2 = y[-13:-1]
        single = predict_params(net, ctx1, input_size=12, scaler=RobustScaler(), loss_fn=g)
        batched = predict_params(net, jnp.stack([ctx1, ctx2]), input_size=12,
                                 scaler=RobustScaler(), loss_fn=g)
        assert batched[0].shape == (2, 4, 2)
        np.testing.assert_allclose(np.asarray(batched[0][0]), np.asarray(single[0]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(batched[1][0]), np.asarray(single[1]), rtol=1e-6)


# =============================================================================
# Model — MLP wrapper (BaseForecaster conformance)
# =============================================================================
from chronax.models.mlp.mlp_model import MLP  # noqa: E402
from chronax.utils import ConformalIntervals  # noqa: E402


def _tiny(**kw):
    base = dict(h=4, input_size=12, hidden_size=16, num_layers=2, max_steps=30,
                windows_batch_size=16, random_seed=0)
    base.update(kw)
    return MLP(**base)


class TestMLPModel:
    def test_fit_predict_shapes_point(self):
        m = _tiny().fit(_make_y())
        out = m.predict(h=4)
        assert set(out) == {"mean"} and out["mean"].shape == (4,)

    def test_gmm_native_interval_keys_and_order(self):
        from chronax.models.mlp.mlp_losses import GMM
        m = _tiny(loss=GMM(n_components=2, num_samples=256)).fit(_make_y())
        out = m.predict(h=4, level=[80, 95])
        assert set(out) == {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
        assert bool(jnp.all(out["lo-95"] <= out["lo-80"]))
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))
        assert bool(jnp.all(out["hi-80"] <= out["hi-95"]))

    def test_gmm_native_ignores_conformal_params(self):
        from chronax.models.mlp.mlp_losses import GMM
        m1 = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_make_y())
        m2 = _tiny(loss=GMM(n_components=2, num_samples=128))
        m2.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m2.fit(_make_y())
        o1, o2 = m1.predict(h=4, level=[80]), m2.predict(h=4, level=[80])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o2[k]))

    def test_point_loss_conformal_path(self):
        m = _tiny()
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        out = m.predict(h=4, level=[80])
        assert {"mean", "lo-80", "hi-80"} <= set(out)
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))

    def test_point_loss_level_without_conformal_raises(self):
        m = _tiny().fit(_make_y())
        with pytest.raises(ValueError, match="conformal_params"):
            m.predict(h=4, level=[80])

    def test_mean_is_analytic_mixture_mean(self):
        from chronax.models.mlp.mlp_losses import GMM
        g = GMM(n_components=3, num_samples=4000)
        m = _tiny(loss=g).fit(_make_y())
        out = m.predict(h=4, level=[80])
        args = predict_params(m.model_, m._contexts, input_size=12,
                              scaler=m._scaler, loss_fn=m._loss_fn)
        np.testing.assert_allclose(np.asarray(out["mean"]),
                                   np.asarray(g.analytic_mean(args)[0]), rtol=1e-6)
        samples = g.sample(args, key=jax.random.PRNGKey(m.random_seed))
        mc = np.asarray(samples.mean(axis=-1))[0]
        sd = np.asarray(samples.std(axis=-1))[0]
        assert np.all(np.abs(mc - np.asarray(out["mean"])) < 4 * sd / np.sqrt(4000))

    def test_predict_level_twice_then_pickle_then_predict_conformal(self):
        m = _tiny()
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_make_y())
        o1 = m.predict(h=4, level=[80])
        o2 = m.predict(h=4, level=[80])
        np.testing.assert_array_equal(np.asarray(o1["lo-80"]), np.asarray(o2["lo-80"]))
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[80])
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o3["mean"]))

    def test_predict_level_twice_then_pickle_then_predict_gmm(self):
        from chronax.models.mlp.mlp_losses import GMM
        m = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_make_y())
        o1 = m.predict(h=4, level=[90])
        o2 = m.predict(h=4, level=[90])
        np.testing.assert_array_equal(np.asarray(o1["hi-90"]), np.asarray(o2["hi-90"]))
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[90])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o3[k]))

    def test_forecast_stateless_matches_fit_predict(self):
        y = _make_y()
        o1 = _tiny().forecast(y, h=4)
        o2 = _tiny().fit(y).predict(h=4)
        np.testing.assert_array_equal(np.asarray(o1["mean"]), np.asarray(o2["mean"]))

    def test_forecast_fitted_values(self):
        y = _make_y()
        out = _tiny().forecast(y, h=4, fitted=True)
        assert out["fitted"].shape == y.shape
        assert bool(jnp.all(jnp.isnan(out["fitted"][:12])))
        assert bool(jnp.all(jnp.isfinite(out["fitted"][12:])))

    def test_forecast_fitted_values_gmm(self):
        from chronax.models.mlp.mlp_losses import GMM
        y = _make_y()
        out = _tiny(loss=GMM(n_components=2)).forecast(y, h=4, fitted=True)
        assert out["fitted"].shape == y.shape
        assert bool(jnp.all(jnp.isfinite(out["fitted"][12:])))

    def test_beats_naive(self):
        t = np.arange(140)
        y = 10 + 0.5 * t + 4 * np.sin(t / 4.0)
        y_train, y_test = jnp.asarray(y[:132], jnp.float32), y[132:]
        m = MLP(h=8, input_size=24, hidden_size=32, max_steps=300,
                windows_batch_size=64, random_seed=0, scaler_type="robust").fit(y_train)
        pred = np.asarray(m.predict(h=8)["mean"])
        mae_mlp = np.mean(np.abs(pred - y_test))
        mae_naive = np.mean(np.abs(float(y_train[-1]) - y_test))
        assert mae_mlp < mae_naive

    def test_futr_exog_fit_predict(self):
        rng = np.random.RandomState(0)
        T, F = 60, 2
        x = jnp.asarray(rng.randn(T, F), jnp.float32)
        m = _tiny().fit(_make_y(T), futr_exog=x)
        out = m.predict(h=4, futr_exog=jnp.asarray(rng.randn(4, F), jnp.float32))
        assert out["mean"].shape == (4,)
        with pytest.raises(ValueError, match="futr_exog"):
            m.predict(h=4)

    def test_rejects_3d_y(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _tiny().fit(jnp.ones((30, 2, 2)))

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _tiny().fit(jnp.ones((12,)))

    def test_h_gt_trained_raises(self):
        m = _tiny().fit(_make_y())
        with pytest.raises(ValueError, match="h"):
            m.predict(h=9)

    def test_default_ctor_matches_nf_defaults(self):
        m = MLP(h=4)
        assert (m.input_size, m.num_layers, m.hidden_size) == (12, 2, 1024)
        assert (m.max_steps, m.learning_rate, m.windows_batch_size) == (1000, 1e-3, 1024)
        assert (m.scaler_type, m.loss, m.random_seed) == ("identity", "mae", 1)


# =============================================================================
# Multivariate (cross-learned N-series) surface
# =============================================================================

def _panel(T=72, seed=0):
    """Three independent series as columns — a plain panel, no hierarchy."""
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    cols = [10 + 0.2 * t + 2 * np.sin(t / 3.0) + 0.3 * rng.randn(T),
            30 + 0.1 * t + 3 * np.cos(t / 5.0) + 0.3 * rng.randn(T),
            5 + 0.05 * t + np.sin(t / 2.0) + 0.2 * rng.randn(T)]
    return jnp.asarray(np.stack(cols, axis=1), jnp.float32)


class TestMLPMultivariate:
    def test_2d_fit_predict_shapes_gmm(self):
        from chronax.models.mlp.mlp_losses import GMM
        m = _tiny(loss=GMM(n_components=2, num_samples=256)).fit(_panel())
        out = m.predict(h=4, level=[80, 95])
        assert set(out) == {"mean", "lo-80", "hi-80", "lo-95", "hi-95"}
        for v in out.values():
            assert v.shape == (4, 3)
        assert bool(jnp.all(out["lo-95"] <= out["lo-80"]))
        assert bool(jnp.all(out["hi-80"] <= out["hi-95"]))

    def test_2d_point_path(self):
        m = _tiny().fit(_panel())
        out = m.predict(h=4)
        assert set(out) == {"mean"} and out["mean"].shape == (4, 3)

    def test_2d_mq_native_path(self):
        from chronax.models.mlp.mlp_losses import MultiQuantileLoss
        m = _tiny(loss=MultiQuantileLoss((0.1, 0.5, 0.9))).fit(_panel())
        out = m.predict(h=4, level=[80])       # 0.10/0.90 are trained quantiles
        assert out["mean"].shape == (4, 3)
        assert out["lo-80"].shape == (4, 3)
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))
        with pytest.raises(ValueError, match="not trained"):
            m.predict(h=4, level=[95])

    def test_rank_rule(self):
        y = _make_y()
        m1 = _tiny().fit(y)
        assert m1.predict(h=4)["mean"].shape == (4,)
        m2 = _tiny().fit(y[:, None])
        assert m2.predict(h=4)["mean"].shape == (4, 1)

    def test_2d_single_column_matches_1d(self):
        y = _make_y()
        p1 = _tiny().fit(y).predict(h=4)["mean"]
        p2 = _tiny().fit(y[:, None]).predict(h=4)["mean"]
        np.testing.assert_array_equal(np.asarray(p1), np.asarray(p2[:, 0]))

    def test_2d_columns_not_scrambled(self):
        # Disjoint value ranges per column: the robust per-window scaler anchors
        # each forecast to its own context median, so a scrambled column order
        # would land forecasts in the wrong range by construction.
        rng = np.random.RandomState(0)
        t = np.arange(60)
        panel = np.stack([1000.0 + np.sin(t / 3.0) + 0.1 * rng.randn(60),
                          5000.0 + np.cos(t / 4.0) + 0.1 * rng.randn(60),
                          9000.0 + np.sin(t / 5.0) + 0.1 * rng.randn(60)], axis=1)
        m = _tiny(scaler_type="robust", max_steps=5).fit(jnp.asarray(panel, jnp.float32))
        mean = np.asarray(m.predict(h=4)["mean"])
        for j, center in enumerate([1000.0, 5000.0, 9000.0]):
            assert np.all(np.abs(mean[:, j] - center) < 500.0)

    def test_2d_point_level_raises(self):
        m = _tiny().fit(_panel())
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        with pytest.raises(ValueError, match="native"):
            m.predict(h=4, level=[80])

    def test_2d_conformity_scores_raises(self):
        m = _tiny()
        m.conformal_params = ConformalIntervals(h=4, n_windows=2)
        m.fit(_panel())
        with pytest.raises(ValueError, match="native"):
            m.conformity_scores(_panel())

    def test_2d_fitted_raises(self):
        with pytest.raises(NotImplementedError, match="fitted"):
            _tiny().forecast(_panel(), h=4, fitted=True)

    def test_2d_shared_futr_exog(self):
        rng = np.random.RandomState(0)
        T, F = 72, 2
        x = jnp.asarray(rng.randn(T, F), jnp.float32)
        m = _tiny().fit(_panel(T), futr_exog=x)
        out = m.predict(h=4, futr_exog=jnp.asarray(rng.randn(4, F), jnp.float32))
        assert out["mean"].shape == (4, 3)
        with pytest.raises(ValueError, match="futr_exog"):
            m.predict(h=4)

    def test_2d_gmm_shared_futr_exog(self):
        # Distribution head + 2-D fit + shared exog: predict_params must
        # broadcast the shared exog window across the batch of contexts.
        from chronax.models.mlp.mlp_losses import GMM
        rng = np.random.RandomState(0)
        T, F = 72, 2
        x = jnp.asarray(rng.randn(T, F), jnp.float32)
        m = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_panel(T), futr_exog=x)
        out = m.predict(h=4, level=[80],
                        futr_exog=jnp.asarray(rng.randn(4, F), jnp.float32))
        assert out["mean"].shape == (4, 3)
        assert bool(jnp.all(out["lo-80"] <= out["hi-80"]))

    def test_2d_fit_is_vmap_traceable(self):
        from chronax.models.mlp.mlp_losses import GMM
        def run(panel):
            m = MLP(h=4, input_size=12, hidden_size=8, max_steps=3,
                    windows_batch_size=16, random_seed=0,
                    loss=GMM(n_components=2, num_samples=64))
            m.fit(panel)
            return m.predict(h=4)["mean"]
        panels = jnp.stack([_panel(seed=0), _panel(seed=1)])
        out = jax.vmap(run)(panels)
        assert out.shape == (2, 4, 3) and bool(jnp.all(jnp.isfinite(out)))

    def test_2d_pickle_roundtrip_and_twice(self):
        from chronax.models.mlp.mlp_losses import GMM
        m = _tiny(loss=GMM(n_components=2, num_samples=128)).fit(_panel())
        o1 = m.predict(h=4, level=[90])
        o2 = m.predict(h=4, level=[90])
        np.testing.assert_array_equal(np.asarray(o1["hi-90"]), np.asarray(o2["hi-90"]))
        m2 = pickle.loads(pickle.dumps(m))
        o3 = m2.predict(h=4, level=[90])
        for k in o1:
            np.testing.assert_array_equal(np.asarray(o1[k]), np.asarray(o3[k]))


# === Refit compile-cache gates (module-level nnx.jit _train_scan) ===
import io as _io
import logging as _logging
import re as _re

_COMPILE_LOG_RE = _re.compile(r"Compiling ")


def _count_compiles(fn):
    """Run fn under jax.log_compiles and return (result, n_xla_compilations)."""
    buf = _io.StringIO()
    handler = _logging.StreamHandler(buf)
    loggers = [
        _logging.getLogger("jax._src.dispatch"),
        _logging.getLogger("jax._src.interpreters.pxla"),
    ]
    with jax.log_compiles(True):
        for lg in loggers:
            lg.addHandler(handler)
        try:
            out = fn()
            jax.block_until_ready(out)
        finally:
            for lg in loggers:
                lg.removeHandler(handler)
    return out, len(_COMPILE_LOG_RE.findall(buf.getvalue()))


def test_count_compiles_canary():
    # Live-fire proof _count_compiles can still see a compile: a fresh jit'd
    # closure always compiles (pjit keys on callable identity), so the counter
    # must report >=1. Guards test_refit_does_not_recompile against going
    # vacuously green if a jax bump renames the log message or logger paths.
    def fresh():
        @jax.jit
        def f(x):
            return x * 4.0 - 3.0
        return f(jnp.ones((2, 5)))

    _, n = _count_compiles(fresh)
    assert n >= 1


def test_refit_does_not_recompile():
    # The training program is a module-level nnx.jit keyed on config graphdefs
    # (value-__eq__ initializers and the memoized optax transform) and operand
    # shapes — a refit AND a fresh same-config instance must hit the cache with
    # zero XLA compiles; fit #2 is counted directly (no uncounted settle call).
    y = jnp.asarray(np.sin(np.arange(40) / 5.0), jnp.float32)
    kw = dict(h=4, input_size=8, max_steps=3, windows_batch_size=4, random_seed=0)
    m = MLP(**kw)
    m.fit(y)                                     # first fit pays the one compile
    _, n_refit = _count_compiles(lambda: (m.fit(y), None)[1])
    assert n_refit == 0
    m2 = MLP(**kw)
    _, n_fresh = _count_compiles(lambda: (m2.fit(y), None)[1])
    assert n_fresh == 0


def test_mqloss_pinball_value_nf_sum():
    # err = y - yhat. Pred zeros, target 2: sum_q max(q*2, (q-1)*2) =
    # 0.2 + 1.0 + 1.8 = 3.0 — neuralforecast's effective reduction (its
    # 1/len(quantiles) factor is dead), matching the trainer's masked branch.
    from chronax.models.mlp.mlp_losses import MultiQuantileLoss

    loss = MultiQuantileLoss((0.1, 0.5, 0.9))
    assert float(loss(jnp.zeros((1, 1, 3)), jnp.array([[2.0]]))) == pytest.approx(3.0)
