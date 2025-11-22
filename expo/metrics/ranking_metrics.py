from __future__ import annotations

from typing import Dict

import numpy as np


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    k = min(k, relevances.shape[-1])
    gains = (2 ** relevances - 1)[:k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains / discounts))

    ideal = np.sort(relevances)[::-1][:k]
    ideal_dcg = float(np.sum((2 ** ideal - 1) / discounts))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def jaccard_at_k(true_indices: np.ndarray, pred_indices: np.ndarray, k: int) -> float:
    true_top = set(true_indices[:k])
    pred_top = set(pred_indices[:k])
    if not true_top and not pred_top:
        return 1.0
    return float(len(true_top & pred_top) / max(len(true_top | pred_top), 1))


def rbo_at_k(true_indices: np.ndarray, pred_indices: np.ndarray, p: float = 0.9, k: int = 50) -> float:
    A = list(true_indices[:k])
    B = list(pred_indices[:k])
    s_A = set()
    s_B = set()
    rbo = 0.0
    for d in range(1, k + 1):
        s_A.add(A[d - 1])
        s_B.add(B[d - 1])
        overlap = len(s_A & s_B) / d
        weight = (1 - p) * (p ** (d - 1))
        rbo += weight * overlap
    return float(rbo)


def rank_metrics_for_profile(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    up_thresh: float,
    down_thresh: float,
    k: int,
    rbo_p: float = 0.9,
) -> Dict[str, float]:
    gains = (np.abs(z_true) >= up_thresh).astype(np.float32)
    ndcg = ndcg_at_k(gains, k=k)
    true_rank = np.argsort(-np.abs(z_true))
    pred_rank = np.argsort(-np.abs(z_pred))
    jac = jaccard_at_k(true_rank, pred_rank, k=k)
    rbo = rbo_at_k(true_rank, pred_rank, p=rbo_p, k=k)
    return {"ndcg": ndcg, "jaccard": jac, "rbo": rbo}
