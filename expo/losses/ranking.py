from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def listnet_distribution(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = scores / temperature
    return F.softmax(scaled, dim=-1)


def two_list_listnet_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    up_thresh: float,
    down_thresh: float,
    temperature: float,
) -> torch.Tensor:
    """Two-list ListNet objective over up- and down-regulated sets."""
    B, G = target.shape
    device = target.device

    up_mask = target >= up_thresh
    down_mask = target <= down_thresh

    loss_terms = []

    for mask in (up_mask, down_mask):
        if not mask.any():
            continue
        idx = mask.view(-1)
        if not idx.any():
            continue

        t = target.view(-1)[idx].view(B, -1)
        p = pred.view(-1)[idx].view(B, -1)

        t_dist = listnet_distribution(t.abs(), temperature=temperature)
        p_dist = listnet_distribution(p, temperature=temperature)
        ce = -(t_dist * (p_dist + 1e-9).log()).sum(dim=-1)
        loss_terms.append(ce.mean())

    if not loss_terms:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return sum(loss_terms) / len(loss_terms)
