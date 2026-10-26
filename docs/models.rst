Models
======

All models are importable from ``chronax.models`` and follow the
``fit()`` → ``predict()`` interface defined by
:class:`~chronax.models.base_forecaster.BaseForecaster`.

Base class
----------

.. automodule:: chronax.models.base_forecaster
   :members:
   :undoc-members:

Multi-series
------------

.. autoclass:: chronax.models.BatchedForecaster
   :members:
   :undoc-members:

Automatic models
----------------

.. autoclass:: chronax.models.AutoARIMA
   :members:
   :undoc-members:

.. autoclass:: chronax.models.AutoETS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.AutoTheta
   :members:
   :undoc-members:

.. autoclass:: chronax.models.AutoTBATS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.AutoMFLES
   :members:
   :undoc-members:

.. autoclass:: chronax.models.AutoCES
   :members:
   :undoc-members:

Exponential smoothing
---------------------

.. autoclass:: chronax.models.Holt
   :members:
   :undoc-members:

.. autoclass:: chronax.models.HoltWinters
   :members:
   :undoc-members:

.. autoclass:: chronax.models.SimpleExponentialSmoothing
   :members:
   :undoc-members:

.. autoclass:: chronax.models.SeasonalExponentialSmoothing
   :members:
   :undoc-members:

.. autoclass:: chronax.models.ETS
   :members:
   :undoc-members:

ARIMA
-----

.. autoclass:: chronax.models.ARIMA
   :members:
   :undoc-members:

Theta
-----

.. autoclass:: chronax.models.Theta
   :members:
   :undoc-members:

Volatility
----------

.. autoclass:: chronax.models.GARCH
   :members:
   :undoc-members:

Decomposition
-------------

.. autoclass:: chronax.models.STL
   :members:
   :undoc-members:

.. autoclass:: chronax.models.MSTL
   :members:
   :undoc-members:

Multi-frequency
---------------

.. autoclass:: chronax.models.TBATS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.MFLES
   :members:
   :undoc-members:

Baseline methods
----------------

.. autoclass:: chronax.models.Naive
   :members:
   :undoc-members:

.. autoclass:: chronax.models.SeasonalNaive
   :members:
   :undoc-members:

.. autoclass:: chronax.models.WindowAverage
   :members:
   :undoc-members:

.. autoclass:: chronax.models.SeasonalWindowAverage
   :members:
   :undoc-members:

.. autoclass:: chronax.models.HistoricAverage
   :members:
   :undoc-members:

.. autoclass:: chronax.models.RandomWalkWithDrift
   :members:
   :undoc-members:

Intermittent demand
-------------------

.. autoclass:: chronax.models.CrostonClassic
   :members:
   :undoc-members:

.. autoclass:: chronax.models.TSB
   :members:
   :undoc-members:

.. autoclass:: chronax.models.ADIDA
   :members:
   :undoc-members:

.. autoclass:: chronax.models.IMAPA
   :members:
   :undoc-members:

Deep learning
-------------

.. autoclass:: chronax.models.iTransformer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.Informer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.TFT
   :members:
   :undoc-members:

.. autoclass:: chronax.models.PatchTST
   :members:
   :undoc-members:

.. autoclass:: chronax.models.Autoformer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.FEDformer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.KAN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.GRU
   :members:
   :undoc-members:

.. autoclass:: chronax.models.DilatedRNN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.XLSTM
   :members:
   :undoc-members:

.. autoclass:: chronax.models.BiTCN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.TCN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.StemGNN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.DeepNPTS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.SOFTS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.SOFTSSharp
   :members:
   :undoc-members:

.. autoclass:: chronax.models.VanillaTransformer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.MLP
   :members:
   :undoc-members:

.. autoclass:: chronax.models.HINT
   :members:
   :undoc-members:

.. autoclass:: chronax.models.NHITS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.TimeMixer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.TimeXer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.DLinear
   :members:
   :undoc-members:

.. autoclass:: chronax.models.NLinear
   :members:
   :undoc-members:

.. autoclass:: chronax.models.XLinear
   :members:
   :undoc-members:

.. autoclass:: chronax.models.RMoK
   :members:
   :undoc-members:

.. autoclass:: chronax.models.NBEATS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.TiDE
   :members:
   :undoc-members:

.. autoclass:: chronax.models.TSMixer
   :members:
   :undoc-members:

.. autoclass:: chronax.models.RNN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.DeepAR
   :members:
   :undoc-members:
