"""
A class for validating and containing the parameters for model cross validation.

Instance Attributes
1. n_windows
2. h
3. method
"""

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
        self.n_windows = n_windows
        self.h = h
        self.method = method
