from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 20,
    title: str = "Calibration curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot mean predicted vs. mean observed in bins."""
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Bin by predicted value
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    bins = np.array_split(np.arange(len(y_pred_sorted)), n_bins)
    pred_means = []
    true_means = []
    for b in bins:
        if len(b) == 0:
            continue
        pred_means.append(float(y_pred_sorted[b].mean()))
        true_means.append(float(y_true_sorted[b].mean()))

    ax.plot(pred_means, true_means, marker="o")
    lims = [
        min(pred_means + true_means),
        max(pred_means + true_means),
    ]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_interval_coverage_curve(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alphas: Sequence[float],
    title: str = "Interval coverage",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot empirical coverage vs nominal for a grid of alphas."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    inside = (y_true >= y_lower) & (y_true <= y_upper)

    nominals = []
    empirical = []
    for alpha in alphas:
        nominal = 1.0 - alpha
        nominals.append(nominal)
        empirical.append(float(inside.mean()))

    ax.plot(nominals, empirical, marker="o")
    lims = [0.0, 1.0]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    fig.tight_layout()
    return ax
