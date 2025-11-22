from __future__ import annotations

from typing import Dict

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error over all entries."""
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root-mean-squared error over all entries."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination (R^2) over all entries."""
    y = target
    y_hat = pred
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation across flattened arrays."""
    x = pred.ravel()
    y = target.ravel()
    if x.size < 2:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = float(np.sqrt((vx ** 2).sum() * (vy ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return float((vx * vy).sum() / denom)


def per_gene_mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-gene MAE assuming arrays of shape (n_profiles, n_genes)."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have same shape")
    return np.mean(np.abs(pred - target), axis=0)


def per_gene_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-gene RMSE assuming arrays of shape (n_profiles, n_genes)."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have same shape")
    return np.sqrt(np.mean((pred - target) ** 2, axis=0))


def aggregate_regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute a small panel of standard regression metrics."""
    return {
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "r2": r2_score(pred, target),
        "pearson": pearson_corr(pred, target),
    }
