from __future__ import annotations

import torch


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    """Element-wise Huber loss.

    This returns the unreduced loss; callers typically take a mean.
    """
    err = pred - target
    abs_err = err.abs()
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
    linear = abs_err - quadratic
    return 0.5 * quadratic**2 + delta * linear
