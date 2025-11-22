from __future__ import annotations

import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    """Predict multiple quantiles from a shared representation."""

    def __init__(self, r: int = 256, out_dim: int = 978, n_quantiles: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(r, out_dim * n_quantiles)
        self.out_dim = out_dim
        self.n_quantiles = n_quantiles

    def forward(self, branch_coef: torch.Tensor, trunk_basis: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = branch_coef * trunk_basis
        out = self.linear(z)
        return out.view(out.size(0), self.n_quantiles, self.out_dim)


class IntervalProjection(nn.Module):
    """Optional non-negative projection for interval widths."""

    def __init__(self) -> None:
        super().__init__()
        self.softplus = nn.Softplus(beta=1.2)

    def forward(self, widths: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.softplus(widths)
