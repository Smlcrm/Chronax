"""
Public Chronax forecasting models namespace.

This package centralizes user-facing forecaster classes so that callers can
import them from a single place (for example, `chronax.models.AutoARIMA`).
"""

from .arima.arima import ARIMA
from .arima.auto_arima import AutoARIMA

from .ets.ets_model import ETS
from .ets.auto_ets import AutoETS

from .tbats.tbats_model import TBATS, AutoTBATS

from .theta.auto_theta import AutoTheta, Theta

from .mfles.mfles import MFLES
from .mfles.auto_mfles import AutoMFLES

from .simple_exponential_smoothing import SimpleExponentialSmoothing
from .seasonal_exponential_smoothing import SeasonalExponentialSmoothing, _seasonal_exponential_smoothing

from .holt import Holt
from .holt_winters import HoltWinters

from .adida import ADIDA
from .croston_classic import CrostonClassic
from .historic_average import HistoricAverage

from .naive import Naive
from .seasonal_naive import SeasonalNaive

from .window_average import WindowAverage
from .seasonal_window_average import SeasonalWindowAverage

from .imapa import IMAPA

from .garch import GARCH

from .mstl import MSTL
from .stl import STL

from .ces import AutoCES

from .tsb import TSB
from .randomWalkWithDrift import RandomWalkWithDrift

# NOTE: GRU is imported lazily (see __getattr__ below) — it hard-pins flax 0.10.x,
# so `import chronax.models` must not force a flax import on callers using other models.

# The autoformer package exports the forecaster as ``AutoformerForecaster``; expose it
# under the public registry name ``Autoformer`` that ``__all__`` advertises.
from .autoformer import AutoformerForecaster as Autoformer

try:
    from .kan import KAN
except ImportError:
    KAN = None

from .itransformer import iTransformer

from .tft import TFT
from .informer import Informer

# PatchTST hard-pins flax 0.10.x (raises ImportError otherwise); guard like KAN so a
# version/availability mismatch degrades to PatchTST=None instead of breaking the namespace.
try:
    from .patchtst import PatchTST
except ImportError:
    PatchTST = None

from .batched_forecaster import BatchedForecaster

from .xlstm import XLSTM

__all__ = [
    "ARIMA",
    "AutoARIMA",
    "ETS",
    "AutoETS",
    "TBATS",
    "AutoTBATS",
    "Theta",
    "AutoTheta",
    "MFLES",
    "AutoMFLES",
    "SimpleExponentialSmoothing",
    "SeasonalExponentialSmoothing",
    "_seasonal_exponential_smoothing",
    "Holt",
    "HoltWinters",
    "ADIDA",
    "CrostonClassic",
    "HistoricAverage",
    "Naive",
    "SeasonalNaive",
    "WindowAverage",
    "SeasonalWindowAverage",
    "IMAPA",
    "GARCH",
    "MSTL",
    "STL",
    "AutoCES",
    "TSB",
    "RandomWalkWithDrift",
    "GRU",
    "Autoformer",
    "iTransformer",
    "TFT",
    "Informer",
    "KAN",
    "PatchTST",
    "BatchedForecaster",
    "XLSTM",
]


def __getattr__(name):
    """Lazily import GRU on first access (PEP 562).

    GRU hard-pins flax 0.10.x (see ``chronax/models/gru/__init__.py``); importing it
    eagerly would make the whole ``chronax.models`` namespace fail to import wherever
    flax is unavailable or version-incompatible. Accessing ``chronax.models.GRU`` (or
    ``from chronax.models import GRU``) triggers the import — and its flax check — only
    when GRU is actually used.
    """
    if name == "GRU":
        from .gru import GRU as _GRU
        return _GRU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

