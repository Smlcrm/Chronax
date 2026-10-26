"""Parameters for conformal prediction interval construction."""

_VALID_METHODS = {"conformal_distribution", "conformal_signed"}

class ConformalIntervals:
    """Configuration for conformal prediction intervals.

    Passed to a forecaster's ``conformal_params`` (or its
    ``prediction_intervals`` alias). ``BaseForecaster.conformity_scores``
    reads it to build ``n_windows`` walk-forward cross-validation windows of
    horizon ``h`` and to pick the interval construction method.

    Args:
        n_windows (int): Number of walk-forward cross-validation windows.
            Must be at least 2. Defaults to 2.
        h (int): Forecast horizon of each window. Defaults to 1.
        method (str): Interval construction method,
            ``"conformal_distribution"`` (symmetric, uses absolute residuals)
            or ``"conformal_signed"`` (asymmetric, uses signed residuals).
            Defaults to ``"conformal_distribution"``.

    Attributes:
        n_windows (int): As passed.
        h (int): As passed.
        method (str): As passed.

    Raises:
        ValueError: If ``n_windows < 2``.
        ValueError: If ``method`` is not one of the two supported names.
    """

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
