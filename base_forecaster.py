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