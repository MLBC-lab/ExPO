from __future__ import annotations

import torch
import torch.nn as nn


class ContextFusionMLP(nn.Module):
    """Fuse molecular and cell context into a single vector."""

    def __init__(self, chem_dim: int, cell_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chem_dim + cell_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
            nn.GELU(),
        )

    def forward(self, chem: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        concat = torch.cat([chem, cell], dim=-1)
        return self.net(concat)


class GatedContextFusion(nn.Module):
    """Alternative fusion that learns a gate between chem and cell embeddings."""

    def __init__(self, chem_dim: int, cell_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(chem_dim + cell_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        self.proj = nn.Linear(chem_dim + cell_dim, out_dim)

    def forward(self, chem: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        stacked = torch.cat([chem, cell], dim=-1)
        logits = self.gate(stacked)
        weights = torch.softmax(logits, dim=-1)
        w_chem = weights[..., 0:1]
        w_cell = weights[..., 1:2]
        fused = torch.cat([w_chem * chem, w_cell * cell], dim=-1)
        return self.proj(fused)
