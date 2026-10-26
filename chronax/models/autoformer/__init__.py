"""Chronax Autoformer — JAX/Flax/Optax univariate forecaster."""
import flax

# Pinned to flax 0.10.x — linen TrainState / Module APIs have churned across
# minor versions. Loud failure here beats silent miscompilation on a newer flax.
if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.Autoformer is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The linen training loop and TrainState wiring will likely need updates "
        f"on a newer flax."
    )

from chronax.models.autoformer.data import (
    RobustScaler,
    batch_generator,
    build_windows,
    create_batch,
    pad_sequence,
    split_train_val_windows,
)
from chronax.models.autoformer.forecaster import Autoformer, AutoformerForecaster
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
    predict_step,
    sample_batch_indices,
    should_stop_early,
    train,
    train_loop,
    train_step,
    train_window_step,
)

__all__ = [
    "Autoformer",
    "AutoformerForecaster",  # deprecated alias
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
