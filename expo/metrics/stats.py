from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None  # type: ignore[assignment]


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(arr[idx].mean())
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return lower, upper


def paired_wilcoxon(x: Sequence[float], y: Sequence[float]) -> float:
    """Return two-sided p-value for a paired Wilcoxon signed-rank test.

    If SciPy is not available, fall back to a simple sign test.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    diff = x_arr - y_arr
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 1.0
    if wilcoxon is not None:
        _, p = wilcoxon(x_arr, y_arr)
        return float(p)
    # sign-test fallback
    n_pos = (nonzero > 0).sum()
    n_neg = (nonzero < 0).sum()
    n = n_pos + n_neg
    # Binomial tail probability at min(n_pos, n_neg)
    k = min(n_pos, n_neg)
    p = 0.0
    for i in range(0, k + 1):
        # C(n, i) * 0.5^n
        from math import comb

        p += comb(n, i) * (0.5 ** n)
    return float(2 * p)


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """Benjamini–Hochberg FDR control.

    Returns a boolean mask of which hypotheses are rejected and the
    associated threshold.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ordered_p = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ordered_p <= thresh
    if not below.any():
        return np.zeros_like(p, dtype=bool), 0.0
    k_max = np.where(below)[0].max()
    p_star = ordered_p[k_max]
    reject = p <= p_star
    return reject, float(p_star)
