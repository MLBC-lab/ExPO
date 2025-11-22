from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    """Container for calibrated prediction intervals and metadata."""

    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    metadata: Dict[str, Any]

    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "center": self.center,
            "metadata": dict(self.metadata),
        }


def stack_predictions(*arrays: np.ndarray) -> np.ndarray:
    """Stack a sequence of prediction arrays along a new model-axis.

    All arrays must have the same shape. The result has shape
    (n_models, *original_shape).
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")
    base_shape = arrays[0].shape
    for a in arrays[1:]:
        if a.shape != base_shape:
            raise ValueError("All arrays must have the same shape to be stacked.")
    return np.stack(arrays, axis=0)


def enforce_monotonic_quantiles(quantile_preds: np.ndarray, axis: int = -1) -> np.ndarray:
    """Project quantile predictions onto the monotone cone along *axis*.

    This is a simple isotonic-like post-processing step that replaces each
    quantile sequence with its cumulative maximum w.r.t. the chosen axis.
    """
    qp = np.asarray(quantile_preds, dtype=float)
    qp_sorted = np.maximum.accumulate(qp, axis=axis)
    return qp_sorted


def center_from_bounds(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return interval centers given lower and upper bounds."""
    return 0.5 * (np.asarray(lower) + np.asarray(upper))


def bounds_from_center_width(center: np.ndarray, width: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return lower/upper bounds given center and total width."""
    c = np.asarray(center)
    w = np.asarray(width)
    half = 0.5 * w
    return c - half, c + half
