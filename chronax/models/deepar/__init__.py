"""DeepAR probabilistic forecasting model (JAX/FLAX/OPTAX implementation).

A complete, NumPy-free implementation of DeepAR with support for:
- LSTM encoder-decoder architecture
- Multiple exogenous variable types (future, static)
- Probabilistic forecasting with Gaussian distribution
- Monte Carlo sampling for uncertainty quantification
"""

__version__ = "1.0.0"

from .model import (
    DeepAR_EncDec,
    nll_gauss,
    quantiles,
    forecast_mc,
    train_model,
)

from .data import (
    create_batch,
    batch_generator,
    align_covariates,
    pad_sequence,
)

from .loss import (
    nll_gaussian,
    nll_gaussian_masked,
    quantile_loss,
)

from .train import (
    TrainState,
    make_loss_fn,
    make_train_step,
    train,
)

from .forecaster import (
    DeepARForecaster,
)

__all__ = [
    # Model core
    "DeepAR_EncDec",
    
    # Loss functions
    "nll_gauss",
    "nll_gaussian",
    "nll_gaussian_masked",
    "quantile_loss",
    
    # Data pipeline
    "create_batch",
    "batch_generator",
    "align_covariates",
    "pad_sequence",
    
    # Training
    "TrainState",
    "make_loss_fn",
    "make_train_step",
    "train",
    
    # Inference
    "forecast_mc",
    "quantiles",
    "train_model",
    
    # High-level API
    "DeepARForecaster",
]
