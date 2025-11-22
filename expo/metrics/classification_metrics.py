from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    from sklearn.metrics import matthews_corrcoef
except Exception:  # pragma: no cover
    matthews_corrcoef = None  # type: ignore[assignment]


def ternarize(z: np.ndarray, up_thresh: float = 1.0, down_thresh: float = -1.0) -> np.ndarray:
    labels = np.zeros_like(z, dtype=np.int8)
    labels[z >= up_thresh] = 1
    labels[z <= down_thresh] = -1
    return labels


def confusion_matrix_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] = (-1, 0, 1),
) -> np.ndarray:
    mapping = {l: i for i, l in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[mapping[int(t)], mapping[int(p)]] += 1
    return C


def mcc_multiclass(C: np.ndarray) -> float:
    """Multiclass MCC from confusion matrix, with a safe fallback."""
    y_true = []
    y_pred = []
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            count = C[i, j]
            y_true.extend([i] * count)
            y_pred.extend([j] * count)
    if not y_true:
        return 0.0
    if matthews_corrcoef is not None:
        return float(matthews_corrcoef(y_true, y_pred))
    # Fallback: compute per-class MCC and average (rough approximation)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for cls in range(C.shape[0]):
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        tn = ((y_true != cls) & (y_pred != cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        if denom == 0:
            scores.append(0.0)
        else:
            scores.append(float((tp * tn - fp * fn) / denom))
    return float(np.mean(scores))
