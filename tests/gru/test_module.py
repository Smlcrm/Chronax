"""Tests for chronax.models.gru.gru_module — init utility, encoder, decoder, GRUNet."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.gru.gru_module import pytorch_uniform_init


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
