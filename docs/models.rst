Models
======

All models are importable from ``chronax.models`` and follow the
``fit()`` → ``predict()`` interface defined by
:class:`~chronax.models.base_forecaster.BaseForecaster`.

Base Class
----------

.. automodule:: chronax.models.base_forecaster
   :members:
   :undoc-members:

Multi-Series
------------

.. autoclass:: chronax.models.BatchedForecaster
   :members:
   :undoc-members:

Automatic Models
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

Exponential Smoothing
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

Multi-Frequency
---------------

.. autoclass:: chronax.models.TBATS
   :members:
   :undoc-members:

.. autoclass:: chronax.models.MFLES
   :members:
   :undoc-members:

Baseline Methods
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

Intermittent Demand
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

Deep Learning
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

.. autoclass:: chronax.models.TCN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.StemGNN
   :members:
   :undoc-members:

.. autoclass:: chronax.models.VanillaTransformer
   :members:
   :undoc-members:
