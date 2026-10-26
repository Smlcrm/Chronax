Chronax
=======

Chronax is a time-series forecasting library in JAX. It provides 59 classical and neural
forecasting models, among them AutoARIMA, AutoETS, AutoTheta, TBATS, MFLES, GARCH, TFT, Informer,
NHITS and PatchTST, behind one ``fit`` / ``predict`` interface. Fitting and forecasting run under
JAX, so the same code runs on CPU, GPU or TPU.

Features
--------

- **JAX-accelerated**: just-in-time (JIT) compiled model fitting and forecasting on CPU, GPU or TPU
- **59 forecasting models**: 28 classical (AutoARIMA, AutoETS, AutoTheta, TBATS, MFLES, GARCH, STL and more) and 31 neural (TFT, Informer, PatchTST, NHITS, TiDE, XLSTM and more)
- **Unified API**: every model follows the same ``fit()`` then ``predict()`` pattern
- **Prediction intervals**: conformal and native interval support
- **NumPy compatible**: accepts and returns standard array types

Installation
------------

From PyPI::

   pip install chronax

Quick start
-----------

All Chronax models follow the same interface: instantiate, ``fit()``, then ``predict()``.

.. code-block:: python

   import jax.numpy as jnp
   from chronax.models import AutoARIMA

   y = jnp.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                   115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])

   model = AutoARIMA(period=12)
   model = model.fit(y)

   forecast = model.predict(h=6, level=[80, 95])
   print("Forecast:", forecast['mean'])

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   models
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
