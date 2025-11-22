from __future__ import annotations

from typing import Any, Callable, Sequence, Type

import numpy as np


class VectorCalibrator:
    """Wrapper that applies a scalar calibrator independently per output dimension."""

    def __init__(
        self,
        base_calibrator_cls: Type[Any],
        n_outputs: int,
        **kwargs: Any,
    ) -> None:
        self.base_calibrator_cls = base_calibrator_cls
        self.n_outputs = int(n_outputs)
        self.kwargs = dict(kwargs)
        self.calibrators: list[Any] = [
            base_calibrator_cls(**self.kwargs) for _ in range(self.n_outputs)
        ]

    def fit(self, y_true: np.ndarray, preds: np.ndarray) -> "VectorCalibrator":
        """Fit one calibrator per output dimension.

        Parameters
        ----------
        y_true:
            Array of shape (n_samples, n_outputs).
        preds:
            Array of shape (n_samples, n_quantiles, n_outputs) or (n_samples, n_outputs).
        """
        y_true = np.asarray(y_true, dtype=float)
        preds = np.asarray(preds, dtype=float)

        if y_true.ndim != 2:
            raise ValueError("y_true must have shape (n_samples, n_outputs)")
        n_samples, n_outputs = y_true.shape
        if n_outputs != self.n_outputs:
            raise ValueError("n_outputs mismatch between data and calibrator.")

        if preds.ndim == 3:
            # (n_samples, n_quantiles, n_outputs)
            for j in range(self.n_outputs):
                self.calibrators[j].fit(y_true[:, j], preds[:, :, j])
        elif preds.ndim == 2:
            for j in range(self.n_outputs):
                self.calibrators[j].fit(y_true[:, j], preds[:, j])
        else:
            raise ValueError("preds must have shape (n, q, d) or (n, d)")
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        preds = np.asarray(preds, dtype=float)
        if preds.ndim == 3:
            n, q, d = preds.shape
            if d != self.n_outputs:
                raise ValueError("Number of outputs in preds does not match calibrator.")
            out = np.empty_like(preds)
            for j in range(self.n_outputs):
                out[:, :, j] = self.calibrators[j].transform(preds[:, :, j])
            return out
        elif preds.ndim == 2:
            n, d = preds.shape
            if d != self.n_outputs:
                raise ValueError("Number of outputs in preds does not match calibrator.")
            out = np.empty_like(preds)
            for j in range(self.n_outputs):
                out[:, j] = self.calibrators[j].transform(preds[:, j])
            return out
        else:
            raise ValueError("preds must have shape (n, q, d) or (n, d)")


def per_dimension_calibrate(
    y_true: np.ndarray,
    preds: np.ndarray,
    base_calibrator_cls: Type[Any],
    **kwargs: Any,
) -> np.ndarray:
    """Convenience function to fit and apply vector calibration in one call."""
    y_true = np.asarray(y_true, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if y_true.ndim != 2:
        raise ValueError("y_true must have shape (n_samples, n_outputs)")
    n_outputs = y_true.shape[1]
    vc = VectorCalibrator(base_calibrator_cls, n_outputs, **kwargs)
    vc.fit(y_true, preds)
    return vc.transform(preds)
