from __future__ import annotations

from typing import Tuple

import numpy as np


def picp(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Prediction interval coverage probability."""
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(inside))


def mpiw(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean prediction interval width."""
    return float(np.mean(y_upper - y_lower))


def risk_coverage_curve(
    errors: np.ndarray,
    interval_widths: np.ndarray,
    coverage_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage curve, returning (coverage, risk)."""
    order = np.argsort(interval_widths)
    sorted_err = np.abs(errors[order])
    n = len(sorted_err)
    coverages = []
    risks = []
    for c in coverage_grid:
        k = max(1, int(round(float(c) * n)))
        coverages.append(float(c))
        risks.append(float(sorted_err[:k].mean()))
    return np.asarray(coverages, dtype=np.float32), np.asarray(risks, dtype=np.float32)


def interval_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float,
) -> float:
    """Interval score (proper scoring rule) for central prediction intervals."""
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    width = y_upper - y_lower
    below = (y_true < y_lower).astype(np.float32)
    above = (y_true > y_upper).astype(np.float32)
    score = width + (2.0 / alpha) * (y_lower - y_true) * below + (2.0 / alpha) * (y_true - y_upper) * above
    return float(np.mean(score))


def summarize_interval_quality(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float,
) -> dict:
    """Return a small dictionary summarizing interval quality."""
    return {
        "alpha": float(alpha),
        "coverage": picp(y_true, y_lower, y_upper),
        "mpiw": mpiw(y_lower, y_upper),
        "interval_score": interval_score(y_true, y_lower, y_upper, alpha),
    }
