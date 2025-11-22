from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DeepONetBranch(nn.Module):
    def __init__(self, in_dim: int, r: int = 256, hidden: Tuple[int, int] = (512, 256)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, r),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(context)


class DeepONetTrunk(nn.Module):
    def __init__(self, in_dim: int, r: int = 256, hidden: Tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, r),
        )

    def forward(self, exposure_feats: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(exposure_feats)


class DeepONetHead(nn.Module):
    def __init__(self, r: int = 256, out_dim: int = 978) -> None:
        super().__init__()
        self.linear = nn.Linear(r, out_dim)

    def forward(self, branch_coef: torch.Tensor, trunk_basis: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = branch_coef * trunk_basis
        return self.linear(z)
