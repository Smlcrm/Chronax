import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import utils

# Plot actual values vs forecast.
def plot_forecast(y, y_hat, y_train=None, ax=None):
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
def plot_forecast_intervals(y, y_hat, lower, upper, ax=None):
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
def plot_forecast_distribution(y, y_samples, ax=None, percentiles=(10, 25, 50, 75, 90, 95)):
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
def plot_forecast_pdf(y_samples, horizon_idx=-1, bins=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(y_samples[:, horizon_idx], kde=True, bins=bins, color="blue", ax=ax)
    ax.set_title("Forecast pdf at Horizon")
    ax.set_xlabel("Forecast value")
    ax.set_ylabel("Density")
    return ax

# Plot the cross validation chained window to evaluate model performance
def plot_chained_window(y, y_preds, horizon=1, ax=None):
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
def acf(x, nlags=40):
    x = jnp.asarray(x)
    x = x - jnp.mean(x)
    result = jnp.correlate(x, x, mode='full')
    acf_vals = result[result.size // 2:] / result[result.size // 2]
    return acf_vals[:nlags + 1]
def plot_acf(x, nlags=40):
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