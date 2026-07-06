"""Chronax FEDformer -- JAX/Flax/Optax univariate forecaster.

Public entry point is :class:`FEDformerForecaster` (exported under the registry
name ``FEDformer`` from ``chronax.models``). The lower-level Flax modules and
math primitives are re-exported for advanced use and testing.
"""

from chronax.models.fedformer.data import (
    RobustScaler,
    batch_generator,
    build_windows,
    create_batch,
    pad_sequence,
    split_train_val_windows,
)
from chronax.models.fedformer.forecaster import FEDformerForecaster
from chronax.models.fedformer.loss import (
    LOSSES,
    LossFn,
    huber,
    mae,
    masked_mae,
    masked_mse,
    mse,
    resolve,
)
from chronax.models.fedformer.model import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FEDformerConfig,
    FEDformerModel,
    FourierBlock,
    FourierCrossAttention,
    MultiHeadProjection,
    SeasonalLayerNorm,
    TokenEmbedding,
    get_frequency_modes,
    moving_avg,
    series_decomp,
)
from chronax.models.fedformer.train import (
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

# Public registry alias: ``from chronax.models import FEDformer``.
FEDformer = FEDformerForecaster

__all__ = [
    # Config + model
    "FEDformerConfig",
    "FEDformerModel",
    "FourierBlock",
    "FourierCrossAttention",
    "MultiHeadProjection",
    "SeasonalLayerNorm",
    "TokenEmbedding",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    # Math primitives
    "moving_avg",
    "series_decomp",
    "get_frequency_modes",
    # High-level forecaster
    "FEDformerForecaster",
    "FEDformer",
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
