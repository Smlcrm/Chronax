"""Tests for chronax.models.gru.gru_module — init utility, encoder, decoder, GRUNet."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.gru.gru_module import GRUEncoder, pytorch_uniform_init


def test_pytorch_uniform_init_bounds():
    """PyTorch nn.GRU/nn.Linear default: Uniform(-1/sqrt(H), 1/sqrt(H))."""
    hidden = 200
    init = pytorch_uniform_init(hidden)
    key = jax.random.PRNGKey(0)
    w = init(key, (hidden, 4 * hidden))
    bound = float(1.0 / jnp.sqrt(jnp.asarray(hidden, dtype=jnp.float32)))
    assert w.shape == (hidden, 4 * hidden)
    assert float(w.min()) >= -bound
    assert float(w.max()) <= bound
    np.testing.assert_allclose(float(w.mean()), 0.0, atol=0.005)


def test_gru_encoder_output_shape():
    encoder = GRUEncoder(
        in_features=1, hidden_size=200, n_layers=2,
        dropout=0.0, rngs=nnx.Rngs(0),
    )
    x = jnp.ones((4, 72, 1))
    out = encoder(x, deterministic=True)
    assert out.shape == (4, 72, 200)


def test_gru_encoder_uses_float32_by_default():
    encoder = GRUEncoder(
        in_features=1, hidden_size=8, n_layers=1,
        dropout=0.0, rngs=nnx.Rngs(0),
    )
    leaves = jax.tree_util.tree_leaves(nnx.state(encoder, nnx.Param))
    assert leaves, "no parameters found"
    for leaf in leaves:
        assert leaf.dtype == jnp.float32, f"expected float32, got {leaf.dtype}"


def test_gru_encoder_documents_bias_parity_with_pytorch():
    """Flax `nnx.GRUCell` fuses biases into a single param on the input
    projection; PyTorch `nn.GRU` has two (b_ih, b_hh). Document the gap so
    accuracy-parity claims are honest about the bounded divergence.
    """
    encoder = GRUEncoder(
        in_features=1, hidden_size=8, n_layers=1,
        dropout=0.0, rngs=nnx.Rngs(0),
    )
    flat = jax.tree_util.tree_leaves_with_path(nnx.state(encoder, nnx.Param))
    bias_count = sum(1 for path, _ in flat if "bias" in str(path).lower())
    assert bias_count == 1, (
        f"Expected 1 bias per cell (Flax convention); got {bias_count}. "
        "PyTorch nn.GRU would have 2 (b_ih, b_hh) — known parity divergence."
    )


def test_gru_encoder_deterministic_with_same_seed():
    x = jnp.ones((2, 10, 1))
    e1 = GRUEncoder(1, 32, 2, 0.0, rngs=nnx.Rngs(42))
    e2 = GRUEncoder(1, 32, 2, 0.0, rngs=nnx.Rngs(42))
    np.testing.assert_allclose(
        e1(x, deterministic=True), e2(x, deterministic=True), rtol=1e-6
    )
