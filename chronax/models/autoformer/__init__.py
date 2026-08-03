"""Chronax Autoformer — JAX/Flax/Optax univariate forecaster."""

from chronax.models.autoformer.data import (
    RobustScaler,
    batch_generator,
    build_windows,
    create_batch,
    pad_sequence,
    split_train_val_windows,
)
from chronax.models.autoformer.forecaster import AutoformerForecaster
from chronax.models.autoformer.loss import (
    LOSSES,
    LossFn,
    huber,
    mae,
    masked_mae,
    masked_mse,
    mse,
    resolve,
)
from chronax.models.autoformer.model import (
    AutoCorrelationLayer,
    AutoformerConfig,
    AutoformerModel,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    SeasonalLayerNorm,
    auto_correlation,
    moving_avg,
    series_decomp,
)
from chronax.models.autoformer.train import (
    TrainState,
    create_train_state,
    eval_step,
    eval_window_step,
    make_lr_schedule,
    sample_batch_indices,
    should_stop_early,
    train_loop,
    train_step,
    train_window_step,
)

# Public alias expected by the chronax.models registry
# (`from .autoformer import Autoformer`), matching the iTransformer naming
# convention where the user-facing class is the bare model name.
Autoformer = AutoformerForecaster

__all__ = [
    "Autoformer",
    # Config + model
    "AutoformerConfig",
    "AutoformerModel",
    "AutoCorrelationLayer",
    "SeasonalLayerNorm",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    # Math primitives
    "moving_avg",
    "series_decomp",
    "auto_correlation",
    # High-level forecaster
    "AutoformerForecaster",
    # Data utilities
    "RobustScaler",
    "build_windows",
    "split_train_val_windows",
    "pad_sequence",
    "create_batch",
    "batch_generator",
    # Loss functions
    "mae",
    "mse",
    "huber",
    "masked_mae",
    "masked_mse",
    "LossFn",
    "LOSSES",
    "resolve",
    # Training
    "TrainState",
    "create_train_state",
    "train_step",
    "eval_step",
    "train_window_step",
    "eval_window_step",
    "train_loop",
    "make_lr_schedule",
    "sample_batch_indices",
    "should_stop_early",
]
