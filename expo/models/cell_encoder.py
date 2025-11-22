from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CellEmbedding(nn.Module):
    def __init__(self, num_cells: int, dim: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_cells, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, cell_indices: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.embedding(cell_indices)


class BasalExpressionProjector(nn.Module):
    def __init__(self, in_dim: int = 978, out_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class CellContextEncoder(nn.Module):
    """Combine cell identity embedding with optional basal expression embedding."""

    def __init__(
        self,
        num_cells: int,
        dim: int = 512,
        use_basal: bool = True,
        basal_dim: int = 978,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cell_embedding = CellEmbedding(num_cells, dim)
        self.use_basal = use_basal
        self.basal_projector: Optional[BasalExpressionProjector]
        if use_basal:
            self.basal_projector = BasalExpressionProjector(basal_dim, dim, dropout=dropout)
        else:
            self.basal_projector = None

    def forward(
        self,
        cell_indices: torch.Tensor,
        basal_expression: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # type: ignore[override]
        cell_emb = self.cell_embedding(cell_indices)
        if self.use_basal and self.basal_projector is not None and basal_expression is not None:
            cell_emb = cell_emb + self.basal_projector(basal_expression)
        return cell_emb
