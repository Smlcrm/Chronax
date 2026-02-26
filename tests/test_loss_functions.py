"""
File: test_loss_functions.py

High-level Purpose:
    Validates correctness and numerical stability of forecasting loss functions
    implemented in `loss_functions.py` through deterministic, edge-case, and
    cross-framework consistency tests.

Problem Solved:
    Ensures metric implementations remain mathematically consistent with
    reference NumPy/TensorFlow calculations and robust under common corner cases
    such as zeros, negatives, and shape broadcasting behavior.

Architectural Role:
    Acts as the regression and verification layer for the metrics subsystem in
    the forecasting codebase, protecting downstream benchmark and model quality
    evaluations from silent metric drift.

Major Classes/Functions:
    - Fixtures: `test_data`, `multi_quantile_data`.
    - Unit tests for all exported deterministic and probabilistic losses.
    - Parametrized tests for dtype behavior and numerical edge cases.

External Dependencies:
    - `pytest`, `numpy`
    - `jax.numpy`
    - `tensorflow` (cross-validation baseline)
    - `loss_functions` module under test

Expected Inputs and Outputs:
    - Input: synthetic fixture arrays and parameterized metric configurations.
    - Output: pytest pass/fail assertions; no returned runtime data.

Example:
    >>> # pytest -q test_loss_functions.py

Assumptions:
    - TensorFlow is available in the test environment for cross-checks.
    - JAX and NumPy produce numerically comparable float outputs under test
      tolerances.

Side Effects:
    - Executes test assertions and can raise failures via pytest.

Author:
    Auto-documented
Date:
    2026-02-21
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import xlogy  # Import xlogy for Tweedie deviance calculation
from typing import Any, Callable, Tuple

# --- New Imports ---
import tensorflow as tf  # Import TensorFlow for cross-validation

# Import all JAX loss functions from the utils package
from chronax.utils.loss_functions import *

# --- Fixtures --- #

@pytest.fixture
def test_data() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Basic test data, now as 4D tensors."""
    # Shape (B, C, H, W) -> (1, 1, 2, 3) = 6 elements
    y_true = jnp.array([[[[10.0, 20.0, 30.0],
                         [40.0, 50.0, 60.0]]]])
    y_pred = jnp.array([[[[12.0, 18.0, 30.0],
                         [45.0, 48.0, 61.0]]]])
    # For scaled losses
    y_seasonal = jnp.array([[[[ 8.0, 22.0, 25.0],
                              [40.0, 55.0, 59.0]]]])
    return y_true, y_pred, y_seasonal

@pytest.fixture
def multi_quantile_data() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Data for multi-quantile losses, now as 4D/5D tensors."""
    # y_true: (B, C, H, W) -> (1, 1, 2, 2) = 4 elements
    y_true = jnp.array([[[[10.0, 20.0],
                         [30.0, 40.0]]]])
    # y_pred_quantiles: (B, C, H, W, Q) -> (1, 1, 2, 2, 3)
    y_pred_quantiles = jnp.array([[[[[ 8.0, 10.0, 12.0],  # Preds for y=10
                                    [15.0, 20.0, 25.0]], # Preds for y=20
                                   [[28.0, 30.0, 32.0],  # Preds for y=30
                                    [35.0, 40.0, 45.0]]]]]) # Preds for y=40
    # quantiles: (Q,) -> (3,)
    quantiles = jnp.array([0.1, 0.5, 0.9])  # Define the quantile levels (Q=3)
    # y_seasonal: (B, C, H, W) -> (1, 1, 2, 2)
    y_seasonal = jnp.array([[[[11.0, 19.0],
                              [33.0, 38.0]]]])
    return y_true, y_pred_quantiles, quantiles, y_seasonal

# --- Tests for Scale-Dependent Errors --- #

def test_mean_absolute_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """
    Validate that mean_absolute_error matches NumPy and TensorFlow references.

    Detailed Description:
        Computes MAE using the module's mean_absolute_error on the fixture
        (y_true, y_pred, y_seasonal). Asserts that the result is close to the
        same value computed with NumPy (np.mean(np.abs(y_true - y_pred))) and
        to the TensorFlow equivalent (tf.reduce_mean(tf.abs(y_true_tf - y_pred_tf))),
        within relative tolerance 1e-6. Ensures the JAX implementation is
        numerically correct and consistent across frameworks.

    Args:
        test_data (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): Fixture
            (y_true, y_pred, y_seasonal); only first two are used for MAE.

    Returns:
        None. Fails via pytest if any assertion fails.

    Raises:
        AssertionError: If JAX result differs from NumPy or TensorFlow beyond rtol.

    Side Effects:
        None. Pure comparison and assert.

    Notes:
        Role: Regression test for the core MAE metric used by scaled and
        relative metrics; guards against implementation drift.
    """
    y_true, y_pred, _ = test_data  # Unpack the fixture data, ignoring the seasonal component
    
    # 1. Calculate JAX implementation
    actual = mean_absolute_error(y_true, y_pred)

    # 2. Calculate NumPy baseline
    expected = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. Calculate TensorFlow cross-validation
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    expected_tf = tf.reduce_mean(tf.abs(y_true_tf - y_pred_tf))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_mean_squared_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate MSE against NumPy and TensorFlow baselines."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = mean_squared_error(y_true, y_pred)

    # 2. NumPy
    expected = np.mean(np.power(np.array(y_true) - np.array(y_pred), 2))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    
    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    expected_tf = tf.reduce_mean(tf.square(y_true_tf - y_pred_tf))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_root_mean_squared_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate RMSE against NumPy and TensorFlow baselines."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = root_mean_squared_error(y_true, y_pred)

    # 2. NumPy
    expected = np.sqrt(np.mean(np.power(np.array(y_true) - np.array(y_pred), 2)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    expected_tf = tf.sqrt(tf.reduce_mean(tf.square(y_true_tf - y_pred_tf)))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_bias(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate element-wise bias computation across frameworks."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = bias(y_true, y_pred)
    
    # 2. NumPy
    expected = np.array(y_pred) - np.array(y_true)  # Calculate the expected value (element-wise subtraction)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    expected_tf = y_pred_tf - y_true_tf
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_cfe(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate cumulative forecast error sequence behavior."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = cfe(y_true, y_pred)

    # 2. NumPy (jnp.cumsum flattens by default, so np.cumsum on flattened array is correct)
    expected = np.cumsum(np.array(y_true) - np.array(y_pred))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow (must flatten manually for cumsum to match)
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    errors_flat = tf.reshape(y_true_tf - y_pred_tf, [-1])
    expected_tf = tf.cumsum(errors_flat)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_pis(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate absolute cumulative forecast error implementation."""
    y_true, y_pred, _ = test_data

    # 1. JAX
    actual = pis(y_true, y_pred)

    # 2. NumPy
    expected = np.abs(np.cumsum(np.array(y_true) - np.array(y_pred)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    
    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    errors_flat = tf.reshape(y_true_tf - y_pred_tf, [-1])
    expected_tf = tf.abs(tf.cumsum(errors_flat))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_spis(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate scaled period-in-stock metric implementation."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = spis(y_true, y_pred)
    
    # 2. NumPy
    pis_val = np.abs(np.cumsum(np.array(y_true) - np.array(y_pred)))  # First, calculate the PIS value using numpy
    mean = np.mean(pis_val)  # Second, calculate the mean of the PIS values
    expected = pis_val / (mean + 1e-8)  # Third, scale the PIS values by the mean (plus epsilon)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    errors_flat = tf.reshape(y_true_tf - y_pred_tf, [-1])
    pis_val_tf = tf.abs(tf.cumsum(errors_flat))
    mean_tf = tf.reduce_mean(pis_val_tf)
    expected_tf = pis_val_tf / (mean_tf + 1e-8)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

# --- Tests for Percentage Errors --- #

def test_mean_absolute_percentage_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate MAPE implementation against reference baselines."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = mean_absolute_percentage_error(y_true, y_pred)

    # 2. NumPy
    expected = np.mean(np.abs(np.array(y_true) - np.array(y_pred)) / np.abs(np.array(y_true)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    expected_tf = tf.reduce_mean(tf.abs(y_true_tf - y_pred_tf) / tf.abs(y_true_tf))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_symmetric_mean_absolute_percentage_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate SMAPE implementation against reference baselines."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    
    # 2. NumPy
    expected = np.mean(np.abs(np.array(y_true) - np.array(y_pred)) / (np.abs(np.array(y_true)) + np.abs(np.array(y_pred))))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    expected_tf = tf.reduce_mean(tf.abs(y_true_tf - y_pred_tf) / (tf.abs(y_true_tf) + tf.abs(y_pred_tf)))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

# --- Tests for Scale-Independent Errors --- #

def test_mean_absolute_scaled_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate MASE implementation against NumPy and TensorFlow."""
    y_true, y_pred, y_seasonal = test_data
    
    # 1. JAX
    actual = mean_absolute_scaled_error(y_true, y_pred, y_seasonal)

    # 2. NumPy
    num = np.abs(np.array(y_true) - np.array(y_pred))  # Calculate the numerator (absolute errors) using numpy
    den = np.mean(np.abs(np.array(y_true) - np.array(y_seasonal)))  # Calculate the denominator (MAE of seasonal baseline)
    expected = np.mean(num / (den + 1e-8))  # Calculate the final expected MASE
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    y_seasonal_tf = tf.constant(np.array(y_seasonal), dtype=tf.float32)
    num_tf = tf.abs(y_true_tf - y_pred_tf)
    den_tf = tf.reduce_mean(tf.abs(y_true_tf - y_seasonal_tf))
    expected_tf = tf.reduce_mean(num_tf / (den_tf + 1e-8))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_relative_mean_absolute_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate relative MAE implementation against reference baselines."""
    y_true, y_pred, y_base = test_data # using y_seasonal as y_base
    
    # 1. JAX
    actual = relative_mean_absolute_error(y_true, y_pred, y_base)
    
    # 2. NumPy
    num = np.abs(np.array(y_true) - np.array(y_pred))  # Calculate the numerator (absolute errors)
    den = np.mean(np.abs(np.array(y_true) - np.array(y_base))) # MAE of base model
    expected = np.mean(num / (den + 1e-8))  # Calculate the final expected RelMAE
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    y_base_tf = tf.constant(np.array(y_base), dtype=tf.float32)
    num_tf = tf.abs(y_true_tf - y_pred_tf)
    den_tf = tf.reduce_mean(tf.abs(y_true_tf - y_base_tf))
    expected_tf = tf.reduce_mean(num_tf / (den_tf + 1e-8))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_normalized_deviation(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate normalized deviation metric implementation."""
    y_true, y_pred, _ = test_data
    
    # 1. JAX
    actual = normalized_deviation(y_true, y_pred)
    
    # 2. NumPy
    num = np.sum(np.abs(np.array(y_true) - np.array(y_pred)))
    den = np.sum(np.array(y_true)) # Matching implementation
    expected = num / (den + 1e-8)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    num_tf = tf.reduce_sum(tf.abs(y_true_tf - y_pred_tf))
    den_tf = tf.reduce_sum(y_true_tf) # Match original implementation
    expected_tf = num_tf / (den_tf + 1e-8)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_mean_squared_scaled_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate MSSE implementation against reference baselines."""
    y_true, y_pred, y_seasonal = test_data
    
    # 1. JAX
    actual = mean_squared_scaled_error(y_true, y_pred, y_seasonal)

    # 2. NumPy
    num = np.power(np.array(y_true) - np.array(y_pred), 2)  # Calculate the numerator (squared errors)
    den = np.mean(np.power(np.array(y_true) - np.array(y_seasonal), 2))  # Calculate the denominator (MSE of seasonal baseline)
    expected = np.mean(num / (den + 1e-8))  # Calculate the final expected MSSE
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    y_seasonal_tf = tf.constant(np.array(y_seasonal), dtype=tf.float32)
    num_tf = tf.square(y_true_tf - y_pred_tf)
    den_tf = tf.reduce_mean(tf.square(y_true_tf - y_seasonal_tf))
    expected_tf = tf.reduce_mean(num_tf / (den_tf + 1e-8))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_root_mean_squared_scaled_error(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate RMSSE implementation against reference baselines."""
    y_true, y_pred, y_seasonal = test_data
    
    # 1. JAX
    actual = root_mean_squared_scaled_error(y_true, y_pred, y_seasonal)

    # 2. NumPy
    num = np.power(np.array(y_true) - np.array(y_pred), 2)  # Calculate the numerator (squared errors)
    den = np.mean(np.power(np.array(y_true) - np.array(y_seasonal), 2))
    expected = np.mean(np.power(num / (den + 1e-8), 0.5))  # Calculate the final expected RMSSE
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    y_seasonal_tf = tf.constant(np.array(y_seasonal), dtype=tf.float32)
    num_tf = tf.square(y_true_tf - y_pred_tf)
    den_tf = tf.reduce_mean(tf.square(y_true_tf - y_seasonal_tf))
    expected_tf = tf.reduce_mean(tf.pow(num_tf / (den_tf + 1e-8), 0.5))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

# --- Tests for Quantile Losses --- #

def test_quantile_loss_basic() -> None:
    """
    Check quantile loss asymmetry for over- vs under-prediction at q=0.1 and q=0.9.

    Detailed Description:
        For a known (y, y_over, y_under) setup, verifies that at q=0.1
        overestimation gives loss 0.9 and underestimation 0.1, and at q=0.9
        the opposite. Cross-validates the same logic with TensorFlow's
        maximum(q*delta, (q-1)*delta) and asserts consistency. Confirms the
        pinball loss penalizes errors in the correct direction per quantile.

    Args:
        None. Uses fixed in-test arrays.

    Returns:
        None. Fails via pytest on assertion error.

    Raises:
        AssertionError: If expected loss values or TF comparison fail.

    Side Effects:
        None.

    Notes:
        Role: Sanity test for quantile loss used in quantile regression and
        probabilistic metrics (MQL, CRPS-style).
    """
    """Test quantile loss with known over/under estimation."""
    y = jnp.array([[1.0, 2.0, 3.0],
                   [1.5, 2.5, 3.5]]).reshape(1, 1, 2, 3)
    y_over = y + 1.0
    y_under = y - 1.0 
    y_tf = tf.constant(np.array(y), dtype=tf.float32)
    y_over_tf = tf.constant(np.array(y_over), dtype=tf.float32)
    y_under_tf = tf.constant(np.array(y_under), dtype=tf.float32)

    # Test q = 0.1
    q = 0.1  # Set quantile level
    actual_over = quantile_loss(y, y_over, q)  # Calculate loss for overestimation
    np.testing.assert_allclose(actual_over, 0.9)  # Assert the result is 0.9
    actual_under = quantile_loss(y, y_under, q)  # Calculate loss for underestimation
    np.testing.assert_allclose(actual_under, 0.1)  # Assert the result is 0.1
    
    # TF cross-validation for q=0.1
    expected_tf_over = tf.reduce_mean(tf.maximum(q * (y_tf - y_over_tf), (q - 1) * (y_tf - y_over_tf)))
    np.testing.assert_allclose(actual_over, expected_tf_over.numpy(), rtol=1e-6)
    expected_tf_under = tf.reduce_mean(tf.maximum(q * (y_tf - y_under_tf), (q - 1) * (y_tf - y_under_tf)))
    np.testing.assert_allclose(actual_under, expected_tf_under.numpy(), rtol=1e-6)

    # Test q = 0.9
    q = 0.9  # Set quantile level
    actual_over = quantile_loss(y, y_over, q)  # Calculate loss for overestimation
    np.testing.assert_allclose(actual_over, 0.1)  # Assert the result is 0.1
    actual_under = quantile_loss(y, y_under, q)  # Calculate loss for underestimation
    np.testing.assert_allclose(actual_under, 0.9)  # Assert the result is 0.9
    
    # TF cross-validation for q=0.9
    expected_tf_over_2 = tf.reduce_mean(tf.maximum(q * (y_tf - y_over_tf), (q - 1) * (y_tf - y_over_tf)))
    np.testing.assert_allclose(actual_over, expected_tf_over_2.numpy(), rtol=1e-6)
    expected_tf_under_2 = tf.reduce_mean(tf.maximum(q * (y_tf - y_under_tf), (q - 1) * (y_tf - y_under_tf)))
    np.testing.assert_allclose(actual_under, expected_tf_under_2.numpy(), rtol=1e-6)


def test_scaled_quantile_loss(test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate scaled quantile loss computation."""
    y_true, y_pred, y_seasonal = test_data
    q = 0.25  # Set a specific quantile level for the test
    
    # 1. JAX
    actual = scaled_quantile_loss(y_true, y_pred, q, y_seasonal)

    # 2. NumPy
    delta = np.array(y_true) - np.array(y_pred)  # Calculate delta (error)
    num_np = np.mean(np.maximum(q * delta, (q - 1) * delta))  # Calculate the numerator (quantile loss) using numpy
    den_np = np.mean(np.abs(np.array(y_true) - np.array(y_seasonal)))  # Calculate the denominator (MAE of seasonal baseline)
    expected = num_np / (den_np + 1e-8)  # Calculate the final expected scaled loss
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    y_seasonal_tf = tf.constant(np.array(y_seasonal), dtype=tf.float32)
    delta_tf = y_true_tf - y_pred_tf
    num_tf = tf.reduce_mean(tf.maximum(q * delta_tf, (q - 1) * delta_tf))
    den_tf = tf.reduce_mean(tf.abs(y_true_tf - y_seasonal_tf))
    expected_tf = num_tf / (den_tf + 1e-8)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_multi_quantile_loss(multi_quantile_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate multi-quantile pinball loss computation."""
    y_true, y_pred_quantiles, quantiles, _ = multi_quantile_data

    # 1. JAX
    actual = multi_quantile_loss(y_true, y_pred_quantiles, quantiles)

    # 2. NumPy (N-D compatible calculation)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred_quantiles)
    q_np = np.array(quantiles)
    # Use broadcasting: (..., 1) - (..., Q) -> (..., Q)
    delta_np = y_true_np[..., None] - y_pred_np
    # Use broadcasting: (Q,) * (..., Q) -> (..., Q)
    loss_np = np.maximum(q_np * delta_np, (q_np - 1) * delta_np)
    expected = np.mean(loss_np) # Mean over all elements
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow (N-D compatible calculation)
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_q_tf = tf.constant(np.array(y_pred_quantiles), dtype=tf.float32)
    q_tf = tf.constant(np.array(quantiles), dtype=tf.float32)
    # y_true_tf: (1,1,2,2) -> (1,1,2,2,1)
    # y_pred_q_tf: (1,1,2,2,3)
    # q_tf: (3,)
    delta_tf = tf.expand_dims(y_true_tf, -1) - y_pred_q_tf
    loss_tf = tf.maximum(q_tf * delta_tf, (q_tf - 1) * delta_tf)
    expected_tf = tf.reduce_mean(loss_tf)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_scaled_multi_quantile_loss(multi_quantile_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate scaled multi-quantile loss computation."""
    y_true, y_pred_quantiles, quantiles, y_seasonal = multi_quantile_data

    # 1. JAX
    actual = scaled_multi_quantile_loss(y_true, y_pred_quantiles, quantiles,y_seasonal)
 
    # 2. NumPy (using JAX components)
    num = multi_quantile_loss(y_true, y_pred_quantiles, quantiles)  # Calculate the numerator (MQL)
    den = mean_absolute_error(y_true, y_seasonal)  # Calculate the denominator (MAE of seasonal baseline)
    # den = mean(abs([10-11, 20-19, 30-33, 40-38])) = mean([1, 1, 3, 2]) = 7/4 = 1.75
    np.testing.assert_allclose(den, 1.75)  # Assert the denominator calculation is correct
    expected = num / (den + 1e-8)  # Calculate the expected scaled MQL
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow (full calculation)
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_q_tf = tf.constant(np.array(y_pred_quantiles), dtype=tf.float32)
    q_tf = tf.constant(np.array(quantiles), dtype=tf.float32)
    y_seasonal_tf = tf.constant(np.array(y_seasonal), dtype=tf.float32)
    
    # MQL (num)
    delta_tf = tf.expand_dims(y_true_tf, -1) - y_pred_q_tf
    loss_tf = tf.maximum(q_tf * delta_tf, (q_tf - 1) * delta_tf)
    num_tf = tf.reduce_mean(loss_tf)
    # MAE (den)
    den_tf = tf.reduce_mean(tf.abs(y_true_tf - y_seasonal_tf))
    
    expected_tf = num_tf / (den_tf + 1e-8)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

# --- Tests for Probabilistic Metrics --- #

def test_coverage() -> None:
    """Validate empirical interval coverage computation."""
    y = jnp.array([[[[10, 20, 30, 40, 50], [60, 70, 80, 90, 100]]]])
    
    # Bounds designed to cover the first 7 points (10 through 70) and exclude the last 3 (80, 90, 100).
    # Coverage logic: 7 True / 10 Total = 0.7
    y_lo = jnp.array([[[[ 5, 15, 25, 35, 45], [55, 65, 75, 85, 95]]]])
    y_hi = jnp.array([[[[15, 25, 35, 45, 55], [65, 75,  1,  1,  1]]]])
    
    # 1. JAX
    actual = coverage(y, y_lo, y_hi)
    np.testing.assert_allclose(actual, 0.7)

    # 2. TensorFlow
    y_tf = tf.constant(np.array(y), dtype=tf.float32)
    y_lo_tf = tf.constant(np.array(y_lo), dtype=tf.float32)
    y_hi_tf = tf.constant(np.array(y_hi), dtype=tf.float32)
    covered_tf = tf.logical_and(y_tf >= y_lo_tf, y_tf <= y_hi_tf)
    expected_tf = tf.reduce_mean(tf.cast(covered_tf, dtype=tf.float32))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_calibration() -> None:
    """Validate quantile calibration fraction computation."""
    y = jnp.array([10, 20, 30, 40, 50]).reshape(1, 1, 1, 5)      # true values
    y_pred = jnp.array([5, 25, 29, 40, 55]).reshape(1, 1, 1, 5)    # predicted quantile values
    # y <= y_pred: [F,  T,  F,  T,  T] -> 3/5 = 0.6
    
    # 1. JAX
    actual = calibration(y, y_pred)
    np.testing.assert_allclose(actual, 0.6)

    # 2. TensorFlow
    y_tf = tf.constant(np.array(y), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    calibrated_tf = y_tf <= y_pred_tf
    expected_tf = tf.reduce_mean(tf.cast(calibrated_tf, dtype=tf.float32))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

def test_scaled_crps(multi_quantile_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> None:
    """Validate scaled CRPS approximation based on multi-quantile loss."""
    y_true, y_pred_quantiles, quantiles, _ = multi_quantile_data

    # 1. JAX
    actual = scaled_crps(y_true, y_pred_quantiles, quantiles)

    # 2. NumPy (using JAX components)
    mql = multi_quantile_loss(y_true, y_pred_quantiles, quantiles)  # Calculate the MQL (part of the CRPS formula)
    counts = y_true.size
    norm = np.sum(np.abs(np.array(y_true)))
    expected = (2 * mql * counts) / (norm + 1e-8)  # Calculate the final expected scaled CRPS
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    # 3. TensorFlow (full calculation)
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_q_tf = tf.constant(np.array(y_pred_quantiles), dtype=tf.float32)
    q_tf = tf.constant(np.array(quantiles), dtype=tf.float32)
    
    # MQL
    delta_tf = tf.expand_dims(y_true_tf, -1) - y_pred_q_tf
    loss_tf = tf.maximum(q_tf * delta_tf, (q_tf - 1) * delta_tf)
    mql_tf = tf.reduce_mean(loss_tf)
    
    counts_tf = tf.cast(tf.size(y_true_tf), dtype=tf.float32)
    norm_tf = tf.reduce_sum(tf.abs(y_true_tf))
    expected_tf = (2.0 * mql_tf * counts_tf) / (norm_tf + 1e-8)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-6)

# --- Tests for Other Losses --- #

@pytest.mark.parametrize(
    "power, expected_loss_fn, tf_loss_fn",
    [
        (0,
         lambda y, y_pred: np.mean((y_pred - y) ** 2),
         lambda y, y_pred: tf.reduce_mean(tf.square(y_pred - y))),
        (1,
         lambda y, y_pred: np.mean(2 * (np.nan_to_num(y * np.log(y / y_pred)) - (y - y_pred))),
         lambda y, y_pred: tf.reduce_mean(2 * (tf.math.xlogy(y, y / y_pred) - (y - y_pred)))),
        (2,
         lambda y, y_pred: np.mean(2 * (np.log(y_pred / y) + y / y_pred - 1)),
         lambda y, y_pred: tf.reduce_mean(2 * (tf.math.log(y_pred / y) + y / y_pred - 1))),
    ]
)

def test_tweedie_deviance(
    test_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    power: float,
    expected_loss_fn: Callable[[np.ndarray, np.ndarray], float],
    tf_loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
) -> None:
    """Validate Tweedie deviance across Gaussian, Poisson, and Gamma cases."""
    y, y_pred, _ = test_data

    # # Optionally add a model/channel dimension if needed
    # if y_pred.ndim == 4:
    #     y_pred = jnp.expand_dims(y_pred, axis=-1)  # Shape -> (B,C,H,W,1)

    # 1. JAX
    actual = tweedie_deviance(y, y_pred, power=power)

    # 2. NumPy
    y_np = np.array(y)
    y_pred_np = np.array(y_pred)
    expected = expected_loss_fn(y_np, y_pred_np)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)

    # 3. TensorFlow
    y_tf = tf.constant(y_np, dtype=tf.float32)
    y_pred_tf = tf.constant(y_pred_np, dtype=tf.float32)
    expected_tf = tf_loss_fn(y_tf, y_pred_tf)
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-4)

def test_tweedie_deviance_zero_handling() -> None:
    """Validate Tweedie Poisson branch behavior when targets contain zeros."""
    # Test power=1 with y=0
    y = jnp.array([1.0, 0.0, 3.0])  # Define true values, including a zero
    y_pred = jnp.array([1.5, 2.0, 2.5])[:, None] # Shape (3, 1)

    # 1. JAX
    actual = tweedie_deviance(y, y_pred, power=1)

    # 2. NumPy
    expected_vals = np.array([
        2 * (1 * np.log(1/1.5) - (1-1.5)),  # Deviance for first item
        2 * 2.0,  # Deviance for second item (y=0)
        2 * (3 * np.log(3/2.5) - (3-2.5))  # Deviance for third item
    ])
    expected = np.mean(expected_vals)  # The final loss is the mean of the deviances
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    
    # 3. TensorFlow (requires manual handling of y=0 case)
    y_tf = tf.constant(np.array(y), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    if y_tf.ndim != y_pred_tf.ndim:
        y_tf = tf.expand_dims(y_tf, axis=-1)
    # Poisson deviance: 2 * (y * log(y / y_pred) - (y - y_pred))
    # tf.math.xlogy handles y=0 for the first term
    term1 = tf.math.xlogy(y_tf, y_tf / y_pred_tf)
    term2 = -(y_tf - y_pred_tf)
    # The special case for y=0 in poisson is: 2 * y_pred
    # The general formula gives: 2 * (0 - (0 - y_pred)) = 2 * y_pred
    # So tf_xlogy handles it correctly.
    expected_tf = tf.reduce_mean(2 * (term1 + term2))
    np.testing.assert_allclose(actual, expected_tf.numpy(), rtol=1e-4)



def test_scaled_error_zero_and_negative_denominator() -> None:
    """
    Ensure scaled error metrics remain defined for degenerate denominators.

    Detailed Description:
        Tests mean_absolute_scaled_error when the seasonal baseline is
        identical to y_true (zero denominator) and when the baseline differs
        in a way that could make denominator components negative. Asserts
        that MASE does not become NaN for the zero case and remains positive
        in the other, guarding against numerical or definitional edge cases in
        production use.

    Args:
        None. Uses small fixed arrays.

    Returns:
        None. Fails if MASE is NaN or non-positive where it should be valid.

    Raises:
        AssertionError: If edge-case behavior is incorrect.

    Side Effects:
        None.

    Notes:
        Role: Edge-case test for scale-independent metrics when baselines
        are poor or data is pathological.
    """
    y_true = jnp.array([10.0, 20.0])
    y_pred = jnp.array([12.0, 18.0])

    # Zero denominator (y_seasonal identical to y_true)
    y_zero = y_true.copy()
    mase_zero = mean_absolute_scaled_error(y_true, y_pred, y_zero)
    assert not jnp.isnan(mase_zero), "MASE should not be NaN for zero denominator"

    # Negative denominator (if differences invert)
    y_neg = jnp.array([15.0, 25.0])
    mase_neg = mean_absolute_scaled_error(y_true, y_pred, y_neg)
    assert mase_neg > 0, "MASE should remain positive even with negative denominator components"


def test_quantile_loss_median_equivalence() -> None:
    """Quantile loss with q=0.5 should behave like MAE/2."""
    y_true = jnp.array([1.0, 2.0, 3.0])
    y_pred = jnp.array([2.0, 1.0, 4.0])
    q = 0.5

    ql = quantile_loss(y_true, y_pred, q)
    mae = mean_absolute_error(y_true, y_pred)
    np.testing.assert_allclose(ql, mae / 2, rtol=1e-6)
    
    # TF cross-validation
    y_true_tf = tf.constant(np.array(y_true), dtype=tf.float32)
    y_pred_tf = tf.constant(np.array(y_pred), dtype=tf.float32)
    ql_tf = tf.reduce_mean(tf.maximum(q * (y_true_tf - y_pred_tf), (q - 1) * (y_true_tf - y_pred_tf)))
    mae_tf = tf.reduce_mean(tf.abs(y_true_tf - y_pred_tf))
    np.testing.assert_allclose(ql_tf.numpy(), mae_tf.numpy() / 2, rtol=1e-6)
    np.testing.assert_allclose(ql, ql_tf.numpy(), rtol=1e-6)


def test_tweedie_deviance_inverse_gaussian() -> None:
    """Tweedie deviance edge case: power > 2 (Inverse Gaussian-like)."""
    y = jnp.array([[[[10.0, 20.0, 30.0],
                     [40.0, 50.0, 60.0]]]])
    y_pred = jnp.array([[[[12.0, 18.0, 30.0],
                         [45.0, 48.0, 61.0]]]])

    power = 3.0  # > 2 inverse Gaussian case

    result = tweedie_deviance(y, y_pred, power=power)
    assert jnp.isfinite(result).all(), "Deviance should be finite for power > 2"


@pytest.mark.parametrize(
    "metric_fn",
    [mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, mean_absolute_scaled_error],
)
def test_percentage_and_scaled_error_parametrized(metric_fn: Callable[..., jnp.ndarray]) -> None:
    """Parametrized test to ensure stability across variety."""
    y_true = jnp.array([10.0, 0.0, -5.0, 20.0])  # includes zero and negative
    y_pred = jnp.array([12.0, 1.0, -4.0, 19.0])
    y_seasonal = jnp.array([9.0, 1.0, -6.0, 22.0])

    # Guard against division by zero or nan
    if "scaled" in metric_fn.__name__:
        val = metric_fn(y_true, y_pred, y_seasonal)
    else:
        val = metric_fn(y_true, y_pred)
    assert jnp.isfinite(val), f"{metric_fn.__name__} returned non-finite value"


@pytest.mark.parametrize(
    "fn",
    [
        mean_absolute_error,
        mean_squared_error,
        root_mean_squared_error,
        mean_absolute_percentage_error,
        symmetric_mean_absolute_percentage_error,
        mean_absolute_scaled_error,
        relative_mean_absolute_error,
    ],
)
def test_dtype_preservation(fn: Callable[..., jnp.ndarray]) -> None:
    """Ensure JAX functions return jnp.ndarray of correct dtype."""
    y_true = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y_pred = jnp.array([1.1, 1.9, 3.2], dtype=jnp.float32)

    # Add seasonal baseline for scaled functions
    y_seasonal = jnp.array([1.0, 2.0, 3.1], dtype=jnp.float32)

    # Call appropriate function signature
    if "scaled" in fn.__name__ or "relative" in fn.__name__:
        result = fn(y_true, y_pred, y_seasonal)
    else:
        result = fn(y_true, y_pred)

    assert isinstance(result, jnp.ndarray), f"{fn.__name__} should return jnp.ndarray"
    assert result.dtype == jnp.float32, f"{fn.__name__} should preserve dtype"