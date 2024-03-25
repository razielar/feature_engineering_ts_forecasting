### Summary of all funcitons used throughout the repo/course
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from statsmodels.tsa.stattools import ccf

############ 6) Lag features
def lag_plot(y: pd.Series, x: pd.Series, lag: int, ax: Axes=None, data_point_size: int=10) -> Axes:
    """Lag plot between target time series (y) and feature time series (x).
    Args:
        y (pd.Series): target time series (e.g. sales).
        x (pd.Series): feature time series (e.g. ad spend).
        lag (int): the amount, we want to lag by.
        ax (Axes, optional): axes object. Defaults to None.
        data_point_size (int, optional): Dot size. Defaults to 10.
    Returns:
        Axes: Lag plot Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=[5,5])
    ax.scatter(y=y, x=x.shift(periods=lag), s=data_point_size)
    ax.set_ylabel("$y_t$")
    ax.set_xlabel(f"$x_{{t-{lag}}}$")
    return ax

def plot_ccf(y: pd.Series, x: pd.Series, lags: int) -> Axes:
    """Plot the Cross Correlation Function (CCF) and its confidence interval (CI)
    Args:
        y (pd.Series): target time series (e.g. sales).
        x (pd.Series): feature time series (e.g. ad spend).
        lags (int): the amount, we want to lag by.
    Returns:
        Axes: CCF plot.
    """
    # Compute CCF and confidence interval
    cross_corrs = ccf(x, y)
    ci = 2 / np.sqrt(len(y))
    # Plot
    fig, ax = plt.subplots(figsize=[10, 5])
    ax.stem(range(0, lags + 1), cross_corrs[: lags + 1]) # We care only of the second column from ccf output.
    ax.fill_between(range(0, lags + 1), ci, y2=-ci, alpha=0.2)
    ax.set_title("Cross-correlation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax

############ 7) Window features
def mad(x):
    """Compute the median absolute deviation: MAD
    Similar to std but using median, instead.
    MAD = median(|xi-median(x|)
    Args:
        x: input value
    Returns: mad
    """
    return np.median(np.abs(x - np.median(x)))

def weighted_mean(x: np.array, weights: np.array) -> np.array:
    """Weighted mean: define our own weighted mean function to pass to .apply(weighted_mean, args=(weights,))
    Args:
        x (np.array): input data
        weights (np.array): weight array
    Returns:
        np.array: weighted mean
    """
    return (weights * x).sum() / weights.sum()

def weighted_std(x: np.array, weights: np.array) -> np.array:
    """Weighted standard deviation (std): define our own weighted std function to pass to .apply(weighted_std, args=(weights,))
    Args:
        x (np.array): input data
        weights (np.array): weight array
    Returns:
        np.array: weighted std
    """
    mean_w = weighted_mean(x, weights)
    var_w = (weights * (x - mean_w)**2).sum() / weights.sum()
    return np.sqrt(var_w)
