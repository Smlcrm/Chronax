"""
Parameters for conformal prediction interval construction.

Instance Attributes
1. n_windows -- number of walk-forward cross-validation windows (>= 2)
2. h -- forecast horizon
3. method -- interval construction method:
   'conformal_distribution' (default, symmetric) or 'conformal_signed' (asymmetric)
"""

_VALID_METHODS = {"conformal_distribution", "conformal_signed"}

class ConformalIntervals:
    # __slots__ = ('n_windows, 'h', 'method')
    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got '{method}'"
            )
        self.n_windows = n_windows
        self.h = h
        self.method = method
