"""Chronax FEDformer -- JAX/Flax/Optax univariate forecaster.

Public entry point is :class:`FEDformer` (exported from ``chronax.models``).
The lower-level Flax modules and math primitives are re-exported for advanced
use and testing. ``FEDformerForecaster`` remains as a deprecated config-based alias.
"""
import flax

# Pinned to flax 0.10.x — linen TrainState / Module APIs have churned across
# minor versions. Loud failure here beats silent miscompilation on a newer flax.
if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.FEDformer is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The linen training loop and TrainState wiring will likely need updates "
        f"on a newer flax."
    )

from chronax.models.fedformer.data import (
    RobustScaler,
    batch_generator,
    build_windows,
    create_batch,
    pad_sequence,
    split_train_val_windows,
)
from chronax.models.fedformer.forecaster import FEDformer, FEDformerForecaster
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
    predict_step,
    sample_batch_indices,
    should_stop_early,
    train,
    train_loop,
    train_step,
    train_window_step,
)

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
    "FEDformer",
    "FEDformerForecaster",  # deprecated alias
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
    "train",
    "predict_step",
    "make_lr_schedule",
    "sample_batch_indices",
    "should_stop_early",
]
