"""
The base_forecaster class defines basic attributes and functionalities for all models in statsforecast.

Instance Attributes:
1. alias; model name, declared in model's __init__
2. conformal_params; a conformal_intervals object, in statsforecast previously named prediction_intervals
3. model_; stores fitted model post-training

Class Attributes:
1. uses_exog; boolean representing model's exogenous variable handling

Methods: 
1. new() has no signature and returns a shallow copy of the object. 
   In the event the original object contains a complex data structure, a shallow copy uses a reference to the same object.
   In contrast, a deep copy duplicates the entire data structure.

   The original implementation of new() in statsforecast copies instance attributes from __dict__.
   Using a dictionary to store instance attributes is flexible, but computationally inefficient.
   __slots__ provides an efficient alternative for storing instance attributes, but is harder to implement. 
   new() is implemented twice, once to handle __dict__ and once to handle __slots__. The latter is commented out.

2. Implementing __repr__ allows users to easily call the name of the model, stored in alias

3. conformity_scores()'s signature consists of two arguments, y and X, both JAX arrays, and returns a 2D JAX array, the model's conformity score on y.
   The second argument, X, is optional to allow exogenous variable handling.
   A model's conformity score is defined by the absolute difference between the forecasted value and the actual value for h forecasted positions and across n_windows.

   There are two notable changes from the original implementation of conformity_scores. 
   (a) native error handling: conformity_scores now natively checks that the conformal_params attribute is not None as well as ensuring an adequate number of samples per window.
       The original implementation used a wrapper function to do so.
        
   (b) vectorization of sequential iteration: vmap enables parallelization of a given function.
       scan_fn is a functional method that calculates the conformity score for a single window.
       The original implementation uses sequential iteration. 

4. add_confidence_intervals()'s signature consists of four arguments: fcst, cs, level, and method. It returns a modified version of fcst. It is a static method.
   fcst is a dictionary that contains a model's forecast results. cs is the 2D JAX array returned by conformity_scores().
   level is a list consisting of either ints or floats, denoting the desired confidence interval. method is a string denoting the conformal method.
   Using the model's conformity score, this method calculates the confidence interval of the model's forecasted values, based on the specified level(s) and conformal method.

Notes:
-  Exogenous variable support is model-specific, not framework-level. 
   The boolean uses_exog must be overriden in the model's implementation.
"""
import jax.numpy as jnp
import utils
from jax import vmap

class BaseForecaster:
    uses_exog = False

    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b
    
    # slots implementation
    # __slots__ = (
    #     'alias', 
    #     'conformal_params',
    #     'model_'
    #     )
    # def new(self):
    #     b = type(self).__new__(type(self))
    #     for cls in type(self).__mro__: # __mro__ returns the chain of classes the object inherits
    #         if hasattr(cls, '__slots__'): # safety check, class level
    #             for attr in cls.__slots__:
    #                 if hasattr(self, attr): # safety check, attribute level
    #                     setattr(b, attr, getattr(self, attr))
    #     return b

    def __repr__(self):
        return self.alias

    def conformity_scores(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
         # safety checks
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
        def scan_fn(i_window):
            train_end = base_train_end + i_window * h
            y_train = y[:train_end]
            y_test = y[train_end : train_end + h]
            if X is not None:
                X_train = X[:train_end]
                X_test = X[train_end : train_end + h]
            else:
                X_train = None
                X_test = None
            fcst_window = self.forecast(h=h, y=y_train, X=X_train, X_future=X_test)  # type: ignore[attr-defined]
            window_scores = jnp.abs(fcst_window['mean'].astype('float32') - y_test)
            return window_scores
        cs = vmap(scan_fn)(jnp.arange(n_windows))
        # self._cs = cs
        return cs

    # calculates confidence intervals at level(s) for forceasted values based on conformity_score
    @staticmethod
    def add_confidence_intervals(
        fcst: dict,
        cs: jnp.ndarray,
        level: list[int | float],
        method: str
        ) -> dict:
        
        # we may consider storing the conformal method specific functions to another file for readability
        def conformal_distribution_intervals(fcst, cs, level):
            level = sorted(level)
            alphas = jnp.array([100 - lv for lv in level])
            cuts_lower = alphas / 200.0
            cuts_upper = 1 - alphas / 200.0
            # reverse lower cuts to match original order
            cuts_lower = cuts_lower[::-1]
            cuts = jnp.concatenate([cuts_lower, cuts_upper])
            mean = fcst["mean"].reshape(1, -1)
            cs_flat = cs.reshape(-1)
            # create forecast paths: mean ± conformity_scores
            scores = jnp.vstack([
                mean - cs_flat.reshape(-1, 1),  # lower paths
                mean + cs_flat.reshape(-1, 1)   # upper paths
            ])
            quantiles = jnp.quantile(
                scores,
                cuts,
                axis=0,
            )
            # column names
            lo_cols = [f"lo-{lv}" for lv in reversed(level)]
            hi_cols = [f"hi-{lv}" for lv in level]
            out_cols = lo_cols + hi_cols
            for i, col in enumerate(out_cols):
                fcst[col] = quantiles[i]
            return fcst
        
        allowed_methods = {"conformal_distribution" : conformal_distribution_intervals}
        if method not in allowed_methods:
            raise ValueError(f"{method} is not valid. Choose from {str(allowed_methods)[1:-1]}")
        
        return allowed_methods[method](fcst, cs, level)


# ---------- Test ----------

if __name__ == "__main__":
    from conformal_intervals import ConformalIntervals
    
    print("=" * 60)
    print("Testing BaseForecaster Class")
    print("=" * 60)
    
    # Create a simple mock forecaster for testing
    class MockForecaster(BaseForecaster):
        """Simple forecaster that always predicts the mean."""
        def __init__(self, alias="MockModel", conformal_params=None):
            self.alias = alias
            self.conformal_params = conformal_params
            self.model_ = {}
        
        def forecast(self, y, h, X=None, X_future=None):
            """Always return the mean of y as forecast."""
            mean_val = jnp.mean(y)
            return {"mean": jnp.full(h, mean_val, dtype=jnp.float32)}
    
    # Test 1: Basic instantiation and __repr__
    print("\n[Test 1] Instantiation and __repr__")
    model = MockForecaster(alias="TestModel")
    assert model.alias == "TestModel", "Alias should be set correctly"
    assert repr(model) == "TestModel", "__repr__ should return alias"
    assert model.uses_exog == False, "Default uses_exog should be False"
    print(f"  Model alias: {model.alias}")
    print(f"  Model repr: {repr(model)}")
    print(f"  uses_exog: {model.uses_exog}")
    print("  ✓ Instantiation and __repr__ work correctly")
    
    # Test 2: new() method (shallow copy)
    print("\n[Test 2] new() method (shallow copy)")
    model_original = MockForecaster(alias="Original")
    model_original.model_ = {"mean": jnp.array([1.0, 2.0, 3.0]), "nested": {"key": "value"}}
    model_copy = model_original.new()
    
    assert model_copy is not model_original, "new() should return a different object"
    assert model_copy.alias == model_original.alias, "Alias should be copied"
    assert model_copy.model_ is model_original.model_, "model_ should be shallow-copied (same reference)"
    
    # Verify shallow copy by modifying nested structure
    model_copy.model_["nested"]["key"] = "modified"
    assert model_original.model_["nested"]["key"] == "modified", "Shallow copy shares nested references"
    
    print(f"  Original id: {id(model_original)}")
    print(f"  Copy id: {id(model_copy)}")
    print(f"  model_ is same reference: {model_copy.model_ is model_original.model_}")
    print("  ✓ new() method works correctly (shallow copy verified)")
    
    # Test 3: conformity_scores with no conformal_params
    print("\n[Test 3] conformity_scores without conformal_params raises ValueError")
    model_no_conf = MockForecaster()
    y_test = jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    
    try:
        model_no_conf.conformity_scores(y_test)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "conformal_params must be initialized" in str(e), "Error message should mention conformal_params"
        print(f"  Correctly raised ValueError: {str(e)[:60]}...")
        print("  ✓ conformity_scores correctly requires conformal_params")
    
    # Test 4: conformity_scores with insufficient samples
    print("\n[Test 4] conformity_scores with insufficient samples raises ValueError")
    ci_small = ConformalIntervals(h=5, n_windows=10)
    model_small = MockForecaster(conformal_params=ci_small)
    y_small = jnp.array([1., 2., 3., 4., 5.])  # Only 5 samples, need at least 2*h+1=11
    
    try:
        model_small.conformity_scores(y_small)
        assert False, "Should have raised ValueError for insufficient samples"
    except ValueError as e:
        assert "at least" in str(e) and "samples per window" in str(e), "Error should mention sample requirement"
        print(f"  Correctly raised ValueError: {str(e)[:80]}...")
        print("  ✓ conformity_scores correctly checks sample size")
    
    # Test 5: conformity_scores with sufficient data
    print("\n[Test 5] conformity_scores with sufficient data")
    ci = ConformalIntervals(h=3, n_windows=4)
    model = MockForecaster(conformal_params=ci)
    # Need at least 2*h+1 = 7 samples, but for 4 windows need more
    # n_windows * h + base_train = 4*3 + base_train >= total samples
    # Let's use 20 samples: base_train=8, test_size=12 (4 windows of 3)
    y_large = jnp.arange(1., 21.)  # 20 samples
    
    cs = model.conformity_scores(y_large)
    
    assert cs.shape == (4, 3), f"Conformity scores should have shape (4, 3), got {cs.shape}"
    assert cs.dtype == jnp.float32, f"Conformity scores should be float32, got {cs.dtype}"
    assert jnp.all(cs >= 0), "Conformity scores should all be non-negative (absolute differences)"
    
    print(f"  Conformity scores shape: {cs.shape}")
    print(f"  Conformity scores dtype: {cs.dtype}")
    print(f"  Sample conformity scores (first window): {cs[0]}")
    print("  ✓ conformity_scores computed successfully")
    
    # Test 6: add_confidence_intervals basic functionality
    print("\n[Test 6] add_confidence_intervals basic functionality")
    fcst = {"mean": jnp.array([5.0, 5.0, 5.0])}
    # Create mock conformity scores: 2 windows, 3 horizon
    cs_mock = jnp.array([[0.5, 1.0, 1.5], [1.0, 0.5, 2.0]])
    level = [80, 95]
    
    result = BaseForecaster.add_confidence_intervals(fcst, cs_mock, level, "conformal_distribution")
    
    assert "mean" in result, "Result should still contain mean"
    assert "lo-80" in result, "Result should contain lo-80"
    assert "hi-80" in result, "Result should contain hi-80"
    assert "lo-95" in result, "Result should contain lo-95"
    assert "hi-95" in result, "Result should contain hi-95"
    assert result["lo-80"].shape == (3,), "Lower bound should have same shape as mean"
    assert result["hi-80"].shape == (3,), "Upper bound should have same shape as mean"
    
    # Check that intervals make sense: lo < mean < hi
    assert jnp.all(result["lo-95"] <= result["lo-80"]), "95% lower should be <= 80% lower"
    assert jnp.all(result["hi-80"] <= result["hi-95"]), "80% upper should be <= 95% upper"
    
    print(f"  Mean: {result['mean']}")
    print(f"  lo-80: {result['lo-80']}")
    print(f"  hi-80: {result['hi-80']}")
    print(f"  lo-95: {result['lo-95']}")
    print(f"  hi-95: {result['hi-95']}")
    print("  ✓ add_confidence_intervals works correctly")
    
    # Test 7: add_confidence_intervals with invalid method
    print("\n[Test 7] add_confidence_intervals with invalid method raises ValueError")
    try:
        BaseForecaster.add_confidence_intervals(fcst, cs_mock, level, "invalid_method")
        assert False, "Should have raised ValueError for invalid method"
    except ValueError as e:
        assert "not valid" in str(e), "Error should mention invalid method"
        print(f"  Correctly raised ValueError: {str(e)[:60]}...")
        print("  ✓ add_confidence_intervals validates method")
    
    # Test 8: add_confidence_intervals with single level
    print("\n[Test 8] add_confidence_intervals with single confidence level")
    fcst_single = {"mean": jnp.array([10.0, 10.0])}
    cs_single = jnp.array([[1.0, 1.0], [1.5, 0.5]])
    level_single = [90]
    
    result_single = BaseForecaster.add_confidence_intervals(
        fcst_single, cs_single, level_single, "conformal_distribution"
    )
    
    assert "lo-90" in result_single, "Result should contain lo-90"
    assert "hi-90" in result_single, "Result should contain hi-90"
    assert len([k for k in result_single.keys() if k.startswith("lo-")]) == 1, "Should have only 1 lower bound"
    assert len([k for k in result_single.keys() if k.startswith("hi-")]) == 1, "Should have only 1 upper bound"
    
    print(f"  Keys in result: {list(result_single.keys())}")
    print("  ✓ Single confidence level works correctly")
    
    # Test 9: Class attribute inheritance
    print("\n[Test 9] Class attribute inheritance")
    class ExogForecaster(BaseForecaster):
        uses_exog = True
    
    model_exog = ExogForecaster()
    model_no_exog = MockForecaster()
    
    assert model_exog.uses_exog == True, "Subclass should be able to override uses_exog"
    assert model_no_exog.uses_exog == False, "Base class default should be False"
    
    print(f"  ExogForecaster.uses_exog: {model_exog.uses_exog}")
    print(f"  MockForecaster.uses_exog: {model_no_exog.uses_exog}")
    print("  ✓ Class attribute inheritance works correctly")
    
    # Test 10: conformity_scores adjusts n_windows for short series
    print("\n[Test 10] conformity_scores adjusts n_windows for short series")
    ci_large = ConformalIntervals(h=2, n_windows=100)  # Request 100 windows
    model_adjust = MockForecaster(conformal_params=ci_large)
    y_short = jnp.arange(1., 13.)  # 12 samples: can fit max (12-1)//2 = 5 windows
    
    cs_adjusted = model_adjust.conformity_scores(y_short)
    
    # Should use 5 windows (max possible), not 100 (requested)
    assert cs_adjusted.shape[0] == 5, f"Should adjust to 5 windows, got {cs_adjusted.shape[0]}"
    assert cs_adjusted.shape[1] == 2, f"Horizon should be 2, got {cs_adjusted.shape[1]}"
    
    print(f"  Requested windows: 100")
    print(f"  Actual windows used: {cs_adjusted.shape[0]}")
    print(f"  Conformity scores shape: {cs_adjusted.shape}")
    print("  ✓ conformity_scores correctly adjusts n_windows for short series")
    
    # Test 11: add_confidence_intervals preserves original fcst keys
    print("\n[Test 11] add_confidence_intervals preserves original forecast keys")
    fcst_extra = {
        "mean": jnp.array([7.0, 8.0, 9.0]),
        "custom_key": "custom_value",
        "another_array": jnp.array([1, 2, 3])
    }
    cs_test = jnp.array([[1.0, 1.0, 1.0]])
    
    result_extra = BaseForecaster.add_confidence_intervals(
        fcst_extra, cs_test, [80], "conformal_distribution"
    )
    
    assert "mean" in result_extra, "Should preserve mean"
    assert "custom_key" in result_extra, "Should preserve custom keys"
    assert "another_array" in result_extra, "Should preserve other arrays"
    assert "lo-80" in result_extra, "Should add lo-80"
    assert "hi-80" in result_extra, "Should add hi-80"
    
    print(f"  Original keys: {list(fcst_extra.keys())}")
    print(f"  Result keys: {list(result_extra.keys())}")
    print("  ✓ add_confidence_intervals preserves all original keys")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ All BaseForecaster tests passed successfully!")
    print("=" * 60)
