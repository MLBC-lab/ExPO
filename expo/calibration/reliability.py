from __future__ import annotations

from typing import Tuple

import numpy as np


def regression_reliability_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, mean_pred, mean_true) for a regression reliability curve."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    indices = np.array_split(np.arange(len(y_pred_sorted)), n_bins)

    bin_centers = []
    mean_pred = []
    mean_true = []
    for idx in indices:
        if len(idx) == 0:
            continue
        bin_centers.append(float(y_pred_sorted[idx].mean()))
        mean_pred.append(float(y_pred_sorted[idx].mean()))
        mean_true.append(float(y_true_sorted[idx].mean()))
    return (
        np.asarray(bin_centers, dtype=float),
        np.asarray(mean_pred, dtype=float),
        np.asarray(mean_true, dtype=float),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Expected calibration error (ECE) for regression.

    This is the average absolute difference between mean prediction and mean
    target within bins of predicted value, weighted by bin frequency.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    indices = np.array_split(np.arange(len(y_pred_sorted)), n_bins)

    total = 0.0
    n = float(len(y_pred_sorted))
    for idx in indices:
        if len(idx) == 0:
            continue
        p_mean = float(y_pred_sorted[idx].mean())
        t_mean = float(y_true_sorted[idx].mean())
        gap = abs(p_mean - t_mean)
        w = len(idx) / n
        total += w * gap
    return float(total)


def root_mean_squared_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Root-mean-squared calibration error over bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    indices = np.array_split(np.arange(len(y_pred_sorted)), n_bins)

    total = 0.0
    n = float(len(y_pred_sorted))
    for idx in indices:
        if len(idx) == 0:
            continue
        p_mean = float(y_pred_sorted[idx].mean())
        t_mean = float(y_true_sorted[idx].mean())
        gap = (p_mean - t_mean) ** 2
        w = len(idx) / n
        total += w * gap
    return float(total ** 0.5)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score for probabilistic classification outputs.

    Parameters
    ----------
    probs:
        Array of predicted probabilities with shape (n_samples, n_classes)
        or (n_samples,) for binary classification.
    labels:
        Integer labels of shape (n_samples,).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim == 1:
        # binary case; treat probs as P(y=1)
        p1 = probs
        y1 = (labels == 1).astype(float)
        return float(np.mean((p1 - y1) ** 2))
    else:
        n, k = probs.shape
        onehot = np.eye(k)[labels]
        return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
