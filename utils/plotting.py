import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import utils
from typing import Optional, Tuple, List, Any

# Plot actual values vs forecast.
def plot_forecast(y: jnp.ndarray, y_hat: jnp.ndarray, y_train: Optional[jnp.ndarray] = None, ax: Optional[Any] = None) -> Any:
    """
    Plots the actual ground truth values against the forecasted values.

    Args:
        y (jnp.ndarray): The actual time series values for the forecast horizon.
        y_hat (jnp.ndarray): The predicted forecast values.
        y_train (Optional[jnp.ndarray], optional): Historical training data to plot before the forecast. Defaults to None.
        ax (Optional[Any], optional): A matplotlib Axes object to plot on. If None, a new figure is created. Defaults to None.

    Returns:
        Any: The matplotlib Axes object containing the generated plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    n = len(y)
    t = jnp.arange(n)
    ax.plot(t, y, label="Actual", color="black", lw=2)
    ax.plot(t, y_hat, label="Forecast", color="blue", lw=2)

    if y_train is not None:
        t_train = jnp.arange(-len(y_train), 0)
        ax.plot(t_train, y_train, color="gray", lw=2, label="Train")

    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.5)
    return ax

# Plot forecast with prediction intervals.
def plot_forecast_intervals(y: jnp.ndarray, y_hat: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray, ax: Optional[Any] = None) -> Any:
    """
    Plots the forecast along with shaded prediction intervals.

    Args:
        y (jnp.ndarray): The actual time series values.
        y_hat (jnp.ndarray): The predicted mean/median forecast values.
        lower (jnp.ndarray): The lower bounds of the prediction interval.
        upper (jnp.ndarray): The upper bounds of the prediction interval.
        ax (Optional[Any], optional): A matplotlib Axes object. Defaults to None.

    Returns:
        Any: The matplotlib Axes object containing the generated plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    n = len(y)
    t = jnp.arange(n)
    ax.plot(t, y, label="Actual", color="black", lw=2)
    ax.plot(t, y_hat, label="Forecast", color="blue", lw=2)
    ax.fill_between(t, lower, upper, color="blue", alpha=0.2, label="Prediction Interval")

    ax.set_title("Forecast with Prediction Intervals")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.5)
    return ax

# Plot the shaded area (fan chart) from forecast distributions.
def plot_forecast_distribution(y: jnp.ndarray, y_samples: jnp.ndarray, ax: Optional[Any] = None, percentiles: Tuple[int, ...] = (10, 25, 50, 75, 90, 95)) -> Any:
    """
    Plots a fan chart showing the forecast distribution using shaded percentile bands.

    Args:
        y (jnp.ndarray): The actual time series values.
        y_samples (jnp.ndarray): A 2D array of simulated forecast samples (shape: [num_samples, horizon]).
        ax (Optional[Any], optional): A matplotlib Axes object. Defaults to None.
        percentiles (Tuple[int, ...], optional): The specific percentiles to plot as shaded regions. Defaults to (10, 25, 50, 75, 90, 95).

    Returns:
        Any: The matplotlib Axes object containing the generated fan chart.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    percentiles = jnp.sort(percentiles)
    t = jnp.arange(len(y_samples[0]))
    for i in range(len(percentiles) // 2):
        lower = jnp.quantile(y_samples, percentiles[i], axis=0)
        upper = jnp.quantile(y_samples, percentiles[-(i + 1)], axis=0)
        alpha = 0.1 + 0.1 * i
        ax.fill_between(t, lower, upper, color="blue", alpha=alpha)

    median = jnp.quantile(y_samples, 50, axis=0)
    ax.plot(t, median, color="blue", lw=2, label="Median Forecast")
    ax.plot(t, y, color="black", lw=2, label="Actual")

    ax.set_title("Forecast Distribution (Fan Chart)")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.5)
    return ax

# Plot the forecast pdf
def plot_forecast_pdf(y_samples: jnp.ndarray, horizon_idx: int = -1, bins: int = 30, ax: Optional[Any] = None) -> Any:
    """
    Plots the Probability Density Function (PDF) of the forecast samples at a specific horizon step.

    Args:
        y_samples (jnp.ndarray): A 2D array of simulated forecast samples.
        horizon_idx (int, optional): The specific index of the forecast horizon to plot. Defaults to -1 (the final step).
        bins (int, optional): The number of bins to use for the histogram. Defaults to 30.
        ax (Optional[Any], optional): A matplotlib Axes object. Defaults to None.

    Returns:
        Any: The matplotlib Axes object containing the plotted PDF.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(y_samples[:, horizon_idx], kde=True, bins=bins, color="blue", ax=ax)
    ax.set_title("Forecast pdf at Horizon")
    ax.set_xlabel("Forecast value")
    ax.set_ylabel("Density")
    return ax

# Plot the cross validation chained window to evaluate model performance
def plot_chained_window(y: jnp.ndarray, y_preds: List[jnp.ndarray], horizon: int = 1, ax: Optional[Any] = None) -> Any:
    """
    Plots sequential cross-validation windows to evaluate rolling model performance.

    Args:
        y (jnp.ndarray): The actual continuous time series values.
        y_preds (List[jnp.ndarray]): A list of forecasted arrays, each representing a CV window.
        horizon (int, optional): The forecast horizon length (currently unused in logic, reserved for future functionality). Defaults to 1.
        ax (Optional[Any], optional): A matplotlib Axes object. Defaults to None.

    Returns:
        Any: The matplotlib Axes object containing the chained cross-validation plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n = len(y)
    ax.plot(jnp.arange(n), y, color="black", lw=2, label="Actual")

    for i, fcst in enumerate(y_preds):
        start = i
        end = start + len(fcst)
        t = jnp.arange(start, end)
        ax.plot(t, fcst, color="blue", alpha=0.4, lw=1.5)

    ax.set_title("Cross Validation Chained Window")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.5)
    ax.legend()
    return ax

# Plot the Autocorrelation Function (ACF) Plot
def acf(x: jnp.ndarray, nlags: int = 40) -> jnp.ndarray:
    """
    Computes the Autocorrelation Function (ACF) array for a given time series.

    Args:
        x (jnp.ndarray): The input time series data.
        nlags (int, optional): The maximum number of lags to compute. Defaults to 40.

    Returns:
        jnp.ndarray: An array containing the autocorrelation coefficients from lag 0 up to nlags.
    """
    x = jnp.asarray(x)
    x = x - jnp.mean(x)
    result = jnp.correlate(x, x, mode='full')
    acf_vals = result[result.size // 2:] / result[result.size // 2]
    return acf_vals[:nlags + 1]

def plot_acf(x: jnp.ndarray, nlags: int = 40) -> None:
    """
    Computes and plots the Autocorrelation Function (ACF) along with 95% statistical significance bounds.

    Args:
        x (jnp.ndarray): The input time series data.
        nlags (int, optional): The number of lags to compute and display. Defaults to 40.

    Returns:
        None: Displays the plot via plt.show().
    """
    acf_vals = acf(x, nlags)
    lags = jnp.arange(len(acf_vals))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, acf_vals, width=0.3, color="blue", edgecolor="black")
    ax.axhline(0, color="black", lw=1)
    ax.axhline(1.96 / jnp.sqrt(len(x)), color="red", ls="--", lw=1)
    ax.axhline(-1.96 / jnp.sqrt(len(x)), color="red", ls="--", lw=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation Function Plot")
    ax.grid(True, alpha=0.5)
    plt.show()