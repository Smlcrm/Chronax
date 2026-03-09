Chronax
=======

A high-performance, JAX-accelerated time-series forecasting library. Chronax provides
a comprehensive suite of classical and modern forecasting models — including AutoARIMA,
AutoETS, AutoTheta, TBATS, MFLES, GARCH, and more — with a unified ``fit`` / ``predict``
interface and hardware-accelerated execution via JAX.

Features
--------

- **JAX-accelerated** — JIT-compiled model fitting and forecasting on CPU, GPU, or TPU
- **20+ forecasting models** including AutoARIMA, AutoETS, AutoTheta, TBATS, MFLES, GARCH, STL, and more
- **Unified API** — every model follows the same ``fit()`` → ``predict()`` pattern
- **Prediction intervals** — built-in conformal and native interval support
- **NumPy compatible** — accepts and returns standard array types

Installation
------------

From PyPI::

   pip install chronax

For local development::

   git clone https://github.com/Smlcrm/ml-library-chronax.git
   cd ml-library-chronax
   pip install -e .

Quick Start
-----------

All Chronax models follow the same interface: instantiate, ``fit()``, then ``predict()``.

.. code-block:: python

   import jax.numpy as jnp
   from chronax.models import AutoARIMA

   y = jnp.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                   115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])

   model = AutoARIMA(season_length=12)
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
