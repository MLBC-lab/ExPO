from __future__ import annotations

from typing import Sequence

import torch


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    diff = target - pred
    return torch.maximum(q * diff, (q - 1) * diff)


def multi_quantile_loss(
    quantile_preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: Sequence[float],
) -> torch.Tensor:
    assert quantile_preds.size(1) == len(quantiles)
    target = target.unsqueeze(1).expand_as(quantile_preds)
    losses = []
    for i, q in enumerate(quantiles):
        losses.append(pinball_loss(quantile_preds[:, i, :], target[:, i, :], q))
    return torch.stack(losses, dim=0).mean()
