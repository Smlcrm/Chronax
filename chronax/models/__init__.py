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

try:
    from .kan import KAN
except ImportError:
    KAN = None

# Autoformer / FEDformer hard-pin flax 0.10.x; guard so a version mismatch
# degrades to None instead of breaking the chronax.models namespace.
try:
    from .autoformer import Autoformer
except ImportError:
    Autoformer = None

try:
    from .fedformer import FEDformer
except ImportError:
    FEDformer = None
# NLinear depends on flax NNX (pinned 0.10.x — see chronax/models/nlinear/__init__.py).
# Guard like KAN/PatchTST so an incompatible flax degrades this model to None instead
# of crashing the entire chronax.models namespace. Note the trade-off: the pin's
# informative ImportError is swallowed here; importing chronax.models.nlinear
# directly surfaces it.
try:
    from .nlinear import NLinear
except ImportError:
    NLinear = None

# XLinear depends on flax NNX (pinned 0.10.x — see chronax/models/xlinear/__init__.py).
# Guard like NLinear so an incompatible flax degrades this model to None instead of
# crashing the entire chronax.models namespace; the pin's informative ImportError is
# swallowed here, but importing chronax.models.xlinear directly surfaces it.
# (Distinct from XLSTM below — different model family.)
try:
    from .xlinear import XLinear
except ImportError:
    XLinear = None
# DLinear depends on flax NNX (pinned 0.10.x — see chronax/models/dlinear/__init__.py).
# Guard like NLinear so an incompatible flax degrades this model to None instead of
# crashing the entire chronax.models namespace; the pin's informative ImportError is
# swallowed here, but importing chronax.models.dlinear directly surfaces it.
try:
    from .dlinear import DLinear
except ImportError:
    DLinear = None

# iTransformer / VanillaTransformer depend on flax NNX (flax 0.10.x). If flax's
# nnx import fails (e.g. an incompatible resolved jax that dropped an API nnx
# needs), guard like PatchTST/BiTCN so the model degrades to None instead of
# crashing the import of the entire chronax.models package (which would block
# every other model, including the non-flax ones the benchmark discovers).
try:
    from .itransformer import iTransformer
except ImportError:
    iTransformer = None

try:
    from .vanillatransformer import VanillaTransformer
except ImportError:
    VanillaTransformer = None

from .tft import TFT
from .informer import Informer

# PatchTST hard-pins flax 0.10.x (raises ImportError otherwise); guard like KAN so a
# version/availability mismatch degrades to PatchTST=None instead of breaking the namespace.
try:
    from .patchtst import PatchTST
except ImportError:
    PatchTST = None

# BiTCN hard-pins flax 0.10.x (raises ImportError otherwise); guard like PatchTST so a
# version/availability mismatch degrades to BiTCN=None instead of breaking the namespace.
try:
    from .bitcn import BiTCN
except ImportError:
    BiTCN = None

# DeepNPTS hard-pins flax 0.10.x (raises ImportError otherwise); guard like BiTCN so a
# version/availability mismatch degrades to DeepNPTS=None instead of breaking the namespace.
try:
    from .deepnpts import DeepNPTS
except ImportError:
    DeepNPTS = None

# SOFTS hard-pins flax 0.10.x (raises ImportError otherwise); guard like DeepNPTS so a
# version/availability mismatch degrades to SOFTS=None instead of breaking the namespace.
try:
    from .softs import SOFTS
except ImportError:
    SOFTS = None

# DilatedRNN hard-pins flax 0.10.x (raises ImportError otherwise); guard like SOFTS so a
# version/availability mismatch degrades to DilatedRNN=None instead of breaking the
# namespace.
try:
    from .dilated_rnn import DilatedRNN
except ImportError:
    DilatedRNN = None
# SOFTSSharp (SOFTS#) hard-pins flax 0.10.x (raises ImportError otherwise); guard like
# SOFTS so a version/availability mismatch degrades to SOFTSSharp=None instead of
# breaking the namespace.
try:
    from .softssharp import SOFTSSharp
except ImportError:
    SOFTSSharp = None

# TCN depends on flax NNX; guard like BiTCN so an incompatible flax degrades it to None
# instead of breaking the namespace.
try:
    from .tcn import TCN
except ImportError:
    TCN = None

# StemGNN depends on flax NNX; guard like TCN so an incompatible flax degrades it to None
# instead of breaking the namespace.
try:
    from .stemgnn import StemGNN
except ImportError:
    StemGNN = None

# MLP depends on flax NNX; guard like TCN so an incompatible flax degrades it to None
# instead of breaking the namespace. GMM (the distribution loss MLP/HINT train with)
# is importable from chronax.models.mlp.
try:
    from .mlp import MLP
except ImportError:
    MLP = None

# HINT wraps an MLP base model (flax NNX); guard like MLP.
try:
    from .hint import HINT
except ImportError:
    HINT = None

# TimeMixer depends on flax NNX; guard like TCN so an incompatible flax degrades
# it to None instead of breaking the namespace.
try:
    from .timemixer import TimeMixer
except ImportError:
    TimeMixer = None

# TimeXer depends on flax NNX; guard like TimeMixer.
try:
    from .timexer import TimeXer
except ImportError:
    TimeXer = None

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
    "FEDformer",
    "iTransformer",
    "VanillaTransformer",
    "TFT",
    "Informer",
    "KAN",
    "NLinear",
    "XLinear",
    "DLinear",
    "PatchTST",
    "BiTCN",
    "DeepNPTS",
    "SOFTS",
    "DilatedRNN",
    "SOFTSSharp",
    "TCN",
    "StemGNN",
    "MLP",
    "HINT",
    "TimeMixer",
    "TimeXer",
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

