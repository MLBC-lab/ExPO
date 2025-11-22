from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import bounds_from_center_width, center_from_bounds


def calibrate_quantiles(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> Dict[str, float]:
    """Compute simple multiplicative scaling factors for predicted intervals.

    This function estimates how much to scale the half-width of the interval so
    that the central point (midpoint between q_lower and q_upper) better
    matches the empirical residuals.
    """
    y_true = np.asarray(y_true, dtype=float)
    q_lower = np.asarray(q_lower, dtype=float)
    q_upper = np.asarray(q_upper, dtype=float)

    center = 0.5 * (q_lower + q_upper)
    widths = np.maximum(q_upper - q_lower, 1e-6)
    errors = np.abs(y_true - center)
    ratio = errors / widths
    scale = float(np.quantile(ratio, 0.9))
    return {"scale": max(scale, 1.0)}


def apply_quantile_scaling(
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (q_lower + q_upper)
    half_width = 0.5 * (q_upper - q_lower) * scale
    return center - half_width, center + half_width


def estimate_residual_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "mad",
) -> float:
    """Estimate a global residual scale parameter.

    Parameters
    ----------
    method:
        "mad" for median absolute deviation, or "std" for standard deviation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    if method == "mad":
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        return 1.4826 * mad  # consistent with Gaussian scale
    elif method == "std":
        return float(np.std(resid))
    else:
        raise ValueError(f"Unknown method: {method!r}")


def calibrate_groupwise_quantiles(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    group_ids: Sequence[int] | np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute per-group quantile scaling factors.

    This is useful when different subsets (e.g., cell lines) exhibit different
    uncertainty characteristics.
    """
    y_true = np.asarray(y_true, dtype=float)
    q_lower = np.asarray(q_lower, dtype=float)
    q_upper = np.asarray(q_upper, dtype=float)
    group_ids = np.asarray(group_ids)

    if y_true.shape != q_lower.shape or y_true.shape != q_upper.shape:
        raise ValueError("y_true, q_lower and q_upper must have identical shapes")

    out: Dict[int, Dict[str, float]] = {}
    for g in np.unique(group_ids):
        mask = group_ids == g
        if not mask.any():
            continue
        params = calibrate_quantiles(y_true[mask], q_lower[mask], q_upper[mask])
        out[int(g)] = params
    return out


@dataclass
class QuantileCalibrator:
    """A lightweight calibrator for conditional quantile predictions.

    The calibrator assumes that each quantile prediction can be adjusted by
    a scalar factor applied to its residuals, estimated from a calibration
    dataset. This is deliberately simple but is enough to capture
    under-/over-dispersion in the raw model outputs.
    """

    quantiles: Sequence[float]
    method: str = "scale"
    scales_: Optional[np.ndarray] = None

    def fit(self, y_true: np.ndarray, quantile_preds: np.ndarray) -> "QuantileCalibrator":
        y_true = np.asarray(y_true, dtype=float)
        q_pred = np.asarray(quantile_preds, dtype=float)

        if q_pred.ndim == 3:
            # (n_samples, n_quantiles, n_targets) -> flatten targets
            n, m, d = q_pred.shape
            q_flat = q_pred.reshape(n, m * d)
            y_flat = np.broadcast_to(y_true, (n, d)).reshape(n, d * 1)
            y_flat = np.repeat(y_flat, m, axis=1)
        elif q_pred.ndim == 2:
            q_flat = q_pred
            y_flat = np.repeat(y_true.reshape(-1, 1), q_flat.shape[1], axis=1)
        else:
            raise ValueError("quantile_preds must have shape (n, q) or (n, q, d)")

        if self.method == "scale":
            centers = np.mean(q_flat, axis=1, keepdims=True)
            widths = np.maximum(np.max(q_flat, axis=1, keepdims=True) - np.min(q_flat, axis=1, keepdims=True), 1e-6)
            errors = np.abs(y_flat - centers)
            ratios = errors / widths
            # For each column (quantile) estimate a typical ratio
            self.scales_ = np.quantile(ratios, 0.9, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")
        return self

    def transform(self, quantile_preds: np.ndarray) -> np.ndarray:
        if self.scales_ is None:
            raise RuntimeError("QuantileCalibrator must be fit before calling transform().")
        q_pred = np.asarray(quantile_preds, dtype=float)
        if q_pred.ndim == 3:
            n, m, d = q_pred.shape
            q_flat = q_pred.reshape(n, m * d)
            centers = np.mean(q_flat, axis=1, keepdims=True)
            deltas = q_flat - centers
            scaled = centers + deltas * self.scales_
            return scaled.reshape(n, m, d)
        elif q_pred.ndim == 2:
            centers = np.mean(q_pred, axis=1, keepdims=True)
            deltas = q_pred - centers
            return centers + deltas * self.scales_
        else:
            raise ValueError("quantile_preds must have shape (n, q) or (n, q, d)")


def multi_quantile_interval_from_center(
    center: np.ndarray,
    residual_scale: float,
    quantiles: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct symmetric intervals around *center* given a residual scale.

    This helper is useful when mapping a point prediction and a scalar scale
    estimate into a grid of quantiles, under a simple homoscedastic model.
    """
    center = np.asarray(center, dtype=float)
    qs = np.asarray(quantiles, dtype=float)
    # For simplicity, treat quantiles as multiples of scale
    # q = 0.5 +/- k * scale with k chosen from standard normal quantiles.
    from scipy.stats import norm  # type: ignore[import]

    z_lo = norm.ppf(qs.min())
    z_hi = norm.ppf(qs.max())
    width = float(residual_scale) * (z_hi - z_lo)
    return bounds_from_center_width(center, width)
