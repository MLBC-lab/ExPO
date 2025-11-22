from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .utils import CalibrationResult, center_from_bounds


def conformal_radius(
    residuals: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Return conformal quantile radius for scalar residuals."""
    residuals = np.abs(residuals)
    q = float(np.quantile(residuals, 1.0 - alpha))
    return q


def conformalize_predictions(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return lower/upper bounds by expanding predictions with conformal radius."""
    r = conformal_radius(residuals, alpha=alpha)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = y_pred - r
    upper = y_pred + r
    return lower, upper


def compute_conformity_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return absolute residuals used as conformity scores for regression."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_true - y_pred)


def inductive_conformal_interval(
    y_calib_true: np.ndarray,
    y_calib_pred: np.ndarray,
    y_test_pred: np.ndarray,
    alpha: float = 0.1,
) -> CalibrationResult:
    """Classic split-conformal interval for regression.

    Parameters
    ----------
    y_calib_true, y_calib_pred:
        Calibration targets and predictions (same shape).
    y_test_pred:
        Predictions for the test points for which we want intervals.
    alpha:
        Miscoverage rate (1 - nominal coverage).
    """
    scores = compute_conformity_scores(y_calib_true, y_calib_pred)
    radius = conformal_radius(scores, alpha=alpha)
    lower, upper = conformalize_predictions(y_test_pred, scores, alpha=alpha)
    center = center_from_bounds(lower, upper)
    meta = {"alpha": float(alpha), "radius": float(radius), "method": "split_conformal"}
    return CalibrationResult(lower=lower, upper=upper, center=center, metadata=meta)


@dataclass
class MultiOutputConformalRegressor:
    """Simple multi-output conformal regressor.

    If *shared_radius* is True, a single scalar radius is estimated across all
    outputs. Otherwise, one radius per output dimension is used.
    """

    alpha: float = 0.1
    shared_radius: bool = True
    radius_: Optional[np.ndarray] = None

    def fit(self, y_true_calib: np.ndarray, y_pred_calib: np.ndarray) -> "MultiOutputConformalRegressor":
        y_true = np.asarray(y_true_calib, dtype=float)
        y_pred = np.asarray(y_pred_calib, dtype=float)
        resid = np.abs(y_true - y_pred)
        if self.shared_radius:
            r = float(np.quantile(resid, 1.0 - self.alpha))
            self.radius_ = np.asarray(r, dtype=float)  # scalar
        else:
            # per-dimension radius
            r = np.quantile(resid, 1.0 - self.alpha, axis=0)
            self.radius_ = np.asarray(r, dtype=float)
        return self

    def predict_interval(self, y_pred_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.radius_ is None:
            raise RuntimeError("MultiOutputConformalRegressor must be fit() before predict_interval().")
        y_pred = np.asarray(y_pred_test, dtype=float)
        if self.shared_radius:
            lower = y_pred - self.radius_
            upper = y_pred + self.radius_
        else:
            # broadcast per-dimension radius
            lower = y_pred - self.radius_
            upper = y_pred + self.radius_
        return lower, upper
