"""BaseForecaster defines the shared interface and common infrastructure for all models in Chronax.

Instance Attributes
-------------------
1. ``alias`` -- model name, declared in model's ``__init__``
2. ``conformal_params`` -- conformal interval configuration (typically a
   ``ConformalIntervals`` instance)
3. ``model_`` -- stores fitted model post-training

Class Attributes
----------------
1. ``uses_exog`` -- boolean representing model's exogenous variable handling

Methods
-------
1. ``new()`` returns a shallow copy of the object, used internally to clone
   a model without mutating state.

2. ``__repr__`` returns the model's alias for easy identification.

3. ``fit(y, X=None)`` [abstractmethod]
   Must be implemented by every subclass. Fits the model to univariate time
   series y, sets ``self.model_``, and returns self. X is optional exogenous input.

4. ``predict(h, X=None, level=None)`` [abstractmethod]
   Must be implemented by every subclass. Generates h-step-ahead forecasts,
   returning a dict with at least {"mean": jnp.ndarray}. Optionally adds
   confidence intervals when level is provided.

5. ``forecast(y, h, X=None, X_future=None, level=None, fitted=False)`` [abstractmethod]
   Must be implemented by every subclass. Stateless fit+predict on y,
   forecasting h steps ahead. Implementations differ significantly across
   models (fitted values, model-specific kwargs, etc.). Subclasses may extend
   the signature with additional optional parameters.

6. ``forward(y, h, X=None, X_future=None, level=None, fitted=False)`` [concrete, overridable]
   Updates the model on new data y and forecasts h steps ahead. Default
   delegates to forecast(). Subclasses with warm-start re-estimation
   (e.g. Holt, HoltWinters, ETS) override this.

7. ``conformity_scores(y, X=None)`` computes signed conformity scores on y
   as a 2D JAX array of shape (n_windows, h). Each score is (actual - forecast)
   for a walk-forward cross-validation window. Positive = underprediction.
   Uses vmap for parallelization over windows.

8. ``add_confidence_intervals(fcst, cs, level, method)`` [staticmethod]
   Adds conformal prediction intervals to a forecast dict using pre-computed
   signed conformity scores. Two methods are available:
   - ``conformal_distribution``: symmetric intervals via mean +/- |scores| (2W paths)
   - ``conformal_signed``: asymmetric intervals via mean + scores (W paths)

Notes
-----
- Conformal interval configuration is owned by ``conformal_params`` (constructor
  argument and instance attribute). Migrated forecasters do not accept a
  ``prediction_intervals`` constructor keyword.

- Exogenous variable support is model-specific, not framework-level.
  The boolean uses_exog must be overridden in the model's implementation.

- Known signature inconsistencies in subclasses (future fixes):
  Naive and RandomWalkWithDrift use reversed (h, y) order in forecast/forward.
  RandomWalkWithDrift.predict is missing the X parameter.
  AutoCES.forecast is missing level and fitted parameters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import lax, vmap

from chronax import utils
from chronax.utils import _add_confidence_intervals


class BaseForecaster(ABC):
    """Abstract base class defining the shared interface for all Chronax forecasting models.

    All models must implement ``fit``, ``predict``, and ``forecast``.
    See module docstring for full attribute and method documentation.
    """

    uses_exog = False

    def new(self) -> "BaseForecaster":
        """Return a shallow copy of this model instance.

        Used internally to clone a model without mutating shared state.

        Returns
        -------
        BaseForecaster
            A new instance of the same type with a shallow-copied ``__dict__``.
        """
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b

    def __repr__(self) -> str:
        """Return the model's alias as its string representation."""
        return self.alias

    @abstractmethod
    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "BaseForecaster":
        """
        Fit the model to univariate time series y.
        Must set ``self.model_`` and return self.
        """

    @abstractmethod
    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int | float] | None = None) -> dict:
        """
        Generate h-step-ahead forecasts.
        Returns a dict with at least {"mean": jnp.ndarray}.
        """

    @abstractmethod
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        """
        Stateless fit+predict on y, forecasting h steps ahead.
        Must return a dict with at least {"mean": jnp.ndarray}.
        Subclasses may extend the signature with model-specific optional parameters.
        """

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        """
        Update the model on new data y and forecast h steps ahead.
        Default delegates to forecast(). Subclasses with warm-start
        behavior (e.g. Holt, HoltWinters, ETS) override this.
        """
        return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)

    def conformity_scores(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute signed conformity scores via walk-forward cross-validation.

        Scores are signed residuals (actual - forecast) across ``n_windows``
        expanding windows of horizon ``h``. Uses ``vmap`` for parallelization.
        The interval construction method determines how scores are used:
        ``conformal_distribution`` takes their absolute value for symmetric
        intervals; ``conformal_signed`` uses them directly for asymmetric intervals.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate time series of observed values.
        X : jnp.ndarray or None, optional
            Exogenous features array, shape ``(n_samples, n_features)``. Only
            used by models where ``uses_exog=True``.

        Returns
        -------
        jnp.ndarray
            2-D array of shape ``(n_windows, h)`` containing signed forecast
            errors for each window and horizon step.

        Raises
        ------
        ValueError
            If ``conformal_params`` is ``None``.
        ValueError
            If the series is too short to form at least 2 windows.
        """
        if self.conformal_params is None:
            raise ValueError(
                "The instance attribute conformal_params must be initialized as a conformal_intervals object."
            )
        n_windows = self.conformal_params.n_windows
        h = self.conformal_params.h
        y = utils.ensure_float(y)
        n_samples = y.size
        # use as many windows as possible for short series
        # subtract 1 for the training set
        n_windows = min(n_windows, (n_samples - 1) // h)
        if n_windows < 2:
            raise ValueError(
                f"Conformal prediction requires at least {2 * h + 1:,} samples per window; series has {n_samples:,}."
            )
        test_size = n_windows * h
        base_train_end = n_samples - test_size
        max_train_size = base_train_end + (n_windows - 1) * h

        # Pad arrays to maximum training size for fixed-size operations
        y_padded = jnp.pad(y, (0, max(0, max_train_size - n_samples)), mode='edge')
        if X is not None:
            X_padded = jnp.pad(X, ((0, max(0, max_train_size - n_samples)), (0, 0)), mode='edge')
        else:
            X_padded = None

        def compute_window_scores(i_window):
            train_end = base_train_end + i_window * h

            # Slice with STATIC maximum size (JAX-compatible), then mask
            # data beyond train_end with edge value to prevent data leakage
            y_train = lax.dynamic_slice(y_padded, (0,), (max_train_size,))
            mask = jnp.arange(max_train_size) < train_end
            y_train = jnp.where(mask, y_train, y_train[train_end - 1])

            y_test = lax.dynamic_slice(y_padded, (train_end,), (h,))

            if X_padded is not None:
                X_train = lax.dynamic_slice(X_padded, (0, 0), (max_train_size, X_padded.shape[1]))
                X_train = jnp.where(mask[:, None], X_train, X_train[train_end - 1])
                X_test = lax.dynamic_slice(X_padded, (train_end, 0), (h, X_padded.shape[1]))
            else:
                X_train = None
                X_test = None

            fcst_window = self.forecast(h=h, y=y_train, X=X_train, X_future=X_test)  # type: ignore[attr-defined]
            window_scores = y_test - fcst_window['mean'].astype('float32')
            return window_scores

        # Use vmap for parallel processing across windows
        cs = vmap(compute_window_scores)(jnp.arange(n_windows))
        # self._cs = cs
        return cs


    @staticmethod
    def add_confidence_intervals(
        fcst: dict,
        cs: jnp.ndarray,
        level: list[int | float],
        method: str,
    ) -> dict:
        """Add conformal prediction intervals to a forecast dict.

        Mutates and returns ``fcst`` with interval columns added in-place,
        keyed as ``"lo-{level}"`` and ``"hi-{level}"`` for each requested level.

        Parameters
        ----------
        fcst : dict
            Forecast dictionary containing at least ``{"mean": jnp.ndarray}``.
        cs : jnp.ndarray
            Signed conformity scores, shape ``(n_windows, h)``.
        level : list of int or float
            Confidence levels, e.g. ``[80, 95]``.
        method : str
            ``"conformal_distribution"`` (symmetric, uses ``|scores|``) or
            ``"conformal_signed"`` (asymmetric, uses raw signed scores).

        Returns
        -------
        dict
            The input ``fcst`` dict with interval arrays added.

        Raises
        ------
        ValueError
            If ``method`` is not a recognised conformal method.
        """
        return _add_confidence_intervals(fcst=fcst, cs=cs, level=level, method=method)


# ---------- Test ----------

# if __name__ == "__main__":
#     from conformal_intervals import ConformalIntervals
    
#     print("=" * 60)
#     print("Testing BaseForecaster Class")
#     print("=" * 60)
    
#     # Create a simple mock forecaster for testing
#     class MockForecaster(BaseForecaster):
#         """Simple forecaster that always predicts the mean."""
#         def __init__(self, alias="MockModel", conformal_params=None):
#             self.alias = alias
#             self.conformal_params = conformal_params
#             self.model_ = {}
        
#         def forecast(self, y, h, X=None, X_future=None):
#             """Always return the mean of y as forecast."""
#             mean_val = jnp.mean(y)
#             return {"mean": jnp.full(h, mean_val, dtype=jnp.float32)}
    
#     # Test 1: Basic instantiation and __repr__
#     print("\n[Test 1] Instantiation and __repr__")
#     model = MockForecaster(alias="TestModel")
#     assert model.alias == "TestModel", "Alias should be set correctly"
#     assert repr(model) == "TestModel", "__repr__ should return alias"
#     assert model.uses_exog == False, "Default uses_exog should be False"
#     print(f"  Model alias: {model.alias}")
#     print(f"  Model repr: {repr(model)}")
#     print(f"  uses_exog: {model.uses_exog}")
#     print("  ✓ Instantiation and __repr__ work correctly")
    
#     # Test 2: new() method (shallow copy)
#     print("\n[Test 2] new() method (shallow copy)")
#     model_original = MockForecaster(alias="Original")
#     model_original.model_ = {"mean": jnp.array([1.0, 2.0, 3.0]), "nested": {"key": "value"}}
#     model_copy = model_original.new()
    
#     assert model_copy is not model_original, "new() should return a different object"
#     assert model_copy.alias == model_original.alias, "Alias should be copied"
#     assert model_copy.model_ is model_original.model_, "model_ should be shallow-copied (same reference)"
    
#     # Verify shallow copy by modifying nested structure
#     model_copy.model_["nested"]["key"] = "modified"
#     assert model_original.model_["nested"]["key"] == "modified", "Shallow copy shares nested references"
    
#     print(f"  Original id: {id(model_original)}")
#     print(f"  Copy id: {id(model_copy)}")
#     print(f"  model_ is same reference: {model_copy.model_ is model_original.model_}")
#     print("  ✓ new() method works correctly (shallow copy verified)")
    
#     # Test 3: conformity_scores with no conformal_params
#     print("\n[Test 3] conformity_scores without conformal_params raises ValueError")
#     model_no_conf = MockForecaster()
#     y_test = jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    
#     try:
#         model_no_conf.conformity_scores(y_test)
#         assert False, "Should have raised ValueError"
#     except ValueError as e:
#         assert "conformal_params must be initialized" in str(e), "Error message should mention conformal_params"
#         print(f"  Correctly raised ValueError: {str(e)[:60]}...")
#         print("  ✓ conformity_scores correctly requires conformal_params")
    
#     # Test 4: conformity_scores with insufficient samples
#     print("\n[Test 4] conformity_scores with insufficient samples raises ValueError")
#     ci_small = ConformalIntervals(h=5, n_windows=10)
#     model_small = MockForecaster(conformal_params=ci_small)
#     y_small = jnp.array([1., 2., 3., 4., 5.])  # Only 5 samples, need at least 2*h+1=11
    
#     try:
#         model_small.conformity_scores(y_small)
#         assert False, "Should have raised ValueError for insufficient samples"
#     except ValueError as e:
#         assert "at least" in str(e) and "samples per window" in str(e), "Error should mention sample requirement"
#         print(f"  Correctly raised ValueError: {str(e)[:80]}...")
#         print("  ✓ conformity_scores correctly checks sample size")
    
#     # Test 5: conformity_scores with sufficient data
#     print("\n[Test 5] conformity_scores with sufficient data")
#     ci = ConformalIntervals(h=3, n_windows=4)
#     model = MockForecaster(conformal_params=ci)
#     # Need at least 2*h+1 = 7 samples, but for 4 windows need more
#     # n_windows * h + base_train = 4*3 + base_train >= total samples
#     # Let's use 20 samples: base_train=8, test_size=12 (4 windows of 3)
#     y_large = jnp.arange(1., 21.)  # 20 samples
    
#     cs = model.conformity_scores(y_large)
    
#     assert cs.shape == (4, 3), f"Conformity scores should have shape (4, 3), got {cs.shape}"
#     assert cs.dtype == jnp.float32, f"Conformity scores should be float32, got {cs.dtype}"
#     assert jnp.all(cs >= 0), "Conformity scores should all be non-negative (absolute differences)"
    
#     print(f"  Conformity scores shape: {cs.shape}")
#     print(f"  Conformity scores dtype: {cs.dtype}")
#     print(f"  Sample conformity scores (first window): {cs[0]}")
#     print("  ✓ conformity_scores computed successfully")
    
#     # Test 6: add_confidence_intervals basic functionality
#     print("\n[Test 6] add_confidence_intervals basic functionality")
#     fcst = {"mean": jnp.array([5.0, 5.0, 5.0])}
#     # Create mock conformity scores: 2 windows, 3 horizon
#     cs_mock = jnp.array([[0.5, 1.0, 1.5], [1.0, 0.5, 2.0]])
#     level = [80, 95]
    
#     result = BaseForecaster.add_confidence_intervals(fcst, cs_mock, level, "conformal_distribution")
    
#     assert "mean" in result, "Result should still contain mean"
#     assert "lo-80" in result, "Result should contain lo-80"
#     assert "hi-80" in result, "Result should contain hi-80"
#     assert "lo-95" in result, "Result should contain lo-95"
#     assert "hi-95" in result, "Result should contain hi-95"
#     assert result["lo-80"].shape == (3,), "Lower bound should have same shape as mean"
#     assert result["hi-80"].shape == (3,), "Upper bound should have same shape as mean"
    
#     # Check that intervals make sense: lo < mean < hi
#     assert jnp.all(result["lo-95"] <= result["lo-80"]), "95% lower should be <= 80% lower"
#     assert jnp.all(result["hi-80"] <= result["hi-95"]), "80% upper should be <= 95% upper"
    
#     print(f"  Mean: {result['mean']}")
#     print(f"  lo-80: {result['lo-80']}")
#     print(f"  hi-80: {result['hi-80']}")
#     print(f"  lo-95: {result['lo-95']}")
#     print(f"  hi-95: {result['hi-95']}")
#     print("  ✓ add_confidence_intervals works correctly")
    
#     # Test 7: add_confidence_intervals with invalid method
#     print("\n[Test 7] add_confidence_intervals with invalid method raises ValueError")
#     try:
#         BaseForecaster.add_confidence_intervals(fcst, cs_mock, level, "invalid_method")
#         assert False, "Should have raised ValueError for invalid method"
#     except ValueError as e:
#         assert "not valid" in str(e), "Error should mention invalid method"
#         print(f"  Correctly raised ValueError: {str(e)[:60]}...")
#         print("  ✓ add_confidence_intervals validates method")
    
#     # Test 8: add_confidence_intervals with single level
#     print("\n[Test 8] add_confidence_intervals with single confidence level")
#     fcst_single = {"mean": jnp.array([10.0, 10.0])}
#     cs_single = jnp.array([[1.0, 1.0], [1.5, 0.5]])
#     level_single = [90]
    
#     result_single = BaseForecaster.add_confidence_intervals(
#         fcst_single, cs_single, level_single, "conformal_distribution"
#     )
    
#     assert "lo-90" in result_single, "Result should contain lo-90"
#     assert "hi-90" in result_single, "Result should contain hi-90"
#     assert len([k for k in result_single.keys() if k.startswith("lo-")]) == 1, "Should have only 1 lower bound"
#     assert len([k for k in result_single.keys() if k.startswith("hi-")]) == 1, "Should have only 1 upper bound"
    
#     print(f"  Keys in result: {list(result_single.keys())}")
#     print("  ✓ Single confidence level works correctly")
    
#     # Test 9: Class attribute inheritance
#     print("\n[Test 9] Class attribute inheritance")
#     class ExogForecaster(BaseForecaster):
#         uses_exog = True
    
#     model_exog = ExogForecaster()
#     model_no_exog = MockForecaster()
    
#     assert model_exog.uses_exog == True, "Subclass should be able to override uses_exog"
#     assert model_no_exog.uses_exog == False, "Base class default should be False"
    
#     print(f"  ExogForecaster.uses_exog: {model_exog.uses_exog}")
#     print(f"  MockForecaster.uses_exog: {model_no_exog.uses_exog}")
#     print("  ✓ Class attribute inheritance works correctly")
    
#     # Test 10: conformity_scores adjusts n_windows for short series
#     print("\n[Test 10] conformity_scores adjusts n_windows for short series")
#     ci_large = ConformalIntervals(h=2, n_windows=100)  # Request 100 windows
#     model_adjust = MockForecaster(conformal_params=ci_large)
#     y_short = jnp.arange(1., 13.)  # 12 samples: can fit max (12-1)//2 = 5 windows
    
#     cs_adjusted = model_adjust.conformity_scores(y_short)
    
#     # Should use 5 windows (max possible), not 100 (requested)
#     assert cs_adjusted.shape[0] == 5, f"Should adjust to 5 windows, got {cs_adjusted.shape[0]}"
#     assert cs_adjusted.shape[1] == 2, f"Horizon should be 2, got {cs_adjusted.shape[1]}"
    
#     print(f"  Requested windows: 100")
#     print(f"  Actual windows used: {cs_adjusted.shape[0]}")
#     print(f"  Conformity scores shape: {cs_adjusted.shape}")
#     print("  ✓ conformity_scores correctly adjusts n_windows for short series")
    
#     # Test 11: add_confidence_intervals preserves original fcst keys
#     print("\n[Test 11] add_confidence_intervals preserves original forecast keys")
#     fcst_extra = {
#         "mean": jnp.array([7.0, 8.0, 9.0]),
#         "custom_key": "custom_value",
#         "another_array": jnp.array([1, 2, 3])
#     }
#     cs_test = jnp.array([[1.0, 1.0, 1.0]])
    
#     result_extra = BaseForecaster.add_confidence_intervals(
#         fcst_extra, cs_test, [80], "conformal_distribution"
#     )
    
#     assert "mean" in result_extra, "Should preserve mean"
#     assert "custom_key" in result_extra, "Should preserve custom keys"
#     assert "another_array" in result_extra, "Should preserve other arrays"
#     assert "lo-80" in result_extra, "Should add lo-80"
#     assert "hi-80" in result_extra, "Should add hi-80"
    
#     print(f"  Original keys: {list(fcst_extra.keys())}")
#     print(f"  Result keys: {list(result_extra.keys())}")
#     print("  ✓ add_confidence_intervals preserves all original keys")
    
#     # Summary
#     print("\n" + "=" * 60)
#     print("✓ All BaseForecaster tests passed successfully!")
#     print("=" * 60)
