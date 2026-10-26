import jax
import jax.numpy as jnp
import pytest

from chronax.models.lstm.lstm import LSTMModel


def test_init_creates_expected_parameter_shapes():
    model = LSTMModel(input_size=3, hidden_size=5, output_size=2)
    params = model.init(jax.random.PRNGKey(0))

    assert params.input_kernel.shape == (3, 20)
    assert params.recurrent_kernel.shape == (5, 20)
    assert params.bias.shape == (20,)
    assert params.output_kernel.shape == (5, 2)
    assert params.output_bias.shape == (2,)


def test_apply_returns_outputs_and_final_state():
    model = LSTMModel(input_size=3, hidden_size=4, output_size=2)
    params = model.init(jax.random.PRNGKey(1))
    inputs = jnp.ones((7, 3))

    outputs, final_state = model.apply(params, inputs)

    assert outputs.shape == (7, 2)
    assert final_state.hidden.shape == (4,)
    assert final_state.cell.shape == (4,)
    assert jnp.all(jnp.isfinite(outputs))


def test_apply_batch_vectorizes_sequences():
    model = LSTMModel(input_size=2, hidden_size=3, output_size=1)
    params = model.init(jax.random.PRNGKey(2))
    inputs = jnp.ones((4, 6, 2))

    outputs, final_state = model.apply_batch(params, inputs)

    assert outputs.shape == (4, 6, 1)
    assert final_state.hidden.shape == (4, 3)
    assert final_state.cell.shape == (4, 3)


def test_apply_is_jittable():
    model = LSTMModel(input_size=2, hidden_size=3, output_size=1)
    params = model.init(jax.random.PRNGKey(3))
    inputs = jnp.ones((5, 2))

    outputs, final_state = jax.jit(model.apply)(params, inputs)

    assert outputs.shape == (5, 1)
    assert final_state.hidden.shape == (3,)


def test_model_dimensions_must_be_positive():
    with pytest.raises(ValueError, match="input_size must be positive"):
        LSTMModel(input_size=0, hidden_size=3, output_size=1)