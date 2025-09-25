"""
A class for validating and containing the parameters for model cross validation.

Instance Attributes
1. n_windows
2. h
3. method
"""

class conformal_intervals:
    # __slots__ = ('n_windows, 'h')
    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        self.n_windows = n_windows
        self.h = h
