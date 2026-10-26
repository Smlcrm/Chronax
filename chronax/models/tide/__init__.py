"""Chronax TiDE — JAX/Flax port of Nixtla NeuralForecast's TiDE model."""

from chronax.models.tide.data import (
    RobustScaler,
    align_covariates,
    batch_generator,
    build_windows,
    create_batch,
    pad_sequence,
    split_train_val_windows,
)
from chronax.models.tide.forecaster import TiDEForecaster
from chronax.models.tide.loss import masked_mae, masked_mse
from chronax.models.tide.model import MLPResidual, TiDE, TiDEConfig
from chronax.models.tide.train import (
    TrainState,
    create_train_state,
    eval_step,
    eval_step_windows,
    eval_step_windows_raw,
    eval_step_windows_std,
    train_loop,
    train_step,
    train_step_windows,
    train_step_windows_raw,
    train_step_windows_std,
)

__all__ = [
    # Model
    "TiDE",
    "TiDEConfig",
    "MLPResidual",
    # Forecaster
    "TiDEForecaster",
    # Loss
    "masked_mae",
    "masked_mse",
    # Data
    "RobustScaler",
    "build_windows",
    "split_train_val_windows",
    "create_batch",
    "batch_generator",
    "pad_sequence",
    "align_covariates",
    # Train
    "TrainState",
    "create_train_state",
    "train_step",
    "eval_step",
    "train_step_windows",
    "eval_step_windows",
    "train_step_windows_raw",
    "eval_step_windows_raw",
    "train_step_windows_std",
    "eval_step_windows_std",
    "train_loop",
]
