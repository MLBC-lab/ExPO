from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


class ConstantBaseline(nn.Module):
    """Predict a learned global constant profile for all inputs."""

    def __init__(self, n_genes: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_genes))

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        bsz = len(batch["smiles"])
        mean = self.bias.unsqueeze(0).expand(bsz, -1)
        return {"mean": mean}


class IdentityBaseline(nn.Module):
    """Baseline that returns the input profile unchanged (for diagnostics)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        expr = batch["expression"]
        if isinstance(expr, np.ndarray):
            expr = torch.from_numpy(expr)
        return {"mean": expr}


class PerCellMeanBaseline(nn.Module):
    """Baseline that predicts the empirical mean profile per cell line.

    This is useful for quantifying how much improvement ExPO gives beyond a
    simple cell-specific average across compounds and exposures.
    """

    def __init__(self, num_cells: int, n_genes: int) -> None:
        super().__init__()
        self.register_buffer("cell_means", torch.zeros(num_cells, n_genes))
        self.register_buffer("counts", torch.zeros(num_cells, dtype=torch.long))

    @torch.no_grad()
    def fit_from_loader(self, loader) -> None:
        """Accumulate means from a data loader of batches with cell_index/expression."""
        sums = torch.zeros_like(self.cell_means)
        counts = torch.zeros_like(self.counts)
        for batch in loader:
            cell_idx = batch["cell_index"].long()
            expr = batch["expression"]
            if isinstance(expr, np.ndarray):
                expr = torch.from_numpy(expr)
            expr = expr.to(sums.device)
            for i in range(expr.size(0)):
                idx = int(cell_idx[i])
                sums[idx] += expr[i]
                counts[idx] += 1
        counts_clamped = counts.clamp_min(1)
        self.cell_means.copy_(sums / counts_clamped.unsqueeze(-1))
        self.counts.copy_(counts)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        cell_idx = batch["cell_index"].long()
        means = self.cell_means[cell_idx]
        return {"mean": means}


class DoseScaledBaseline(nn.Module):
    """Baseline that scales a learned mean profile by a simple dose-dependent factor."""

    def __init__(self, n_genes: int, alpha: float = 0.1) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros(n_genes))
        self.alpha = float(alpha)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        dose = batch.get("dose")
        if dose is None:
            # fall back to constant baseline behaviour
            bsz = len(batch["smiles"])
            mean = self.base.unsqueeze(0).expand(bsz, -1)
            return {"mean": mean}
        if isinstance(dose, np.ndarray):
            d = torch.from_numpy(dose).float()
        else:
            d = dose.float()
        d = d.unsqueeze(-1)
        scale = 1.0 + self.alpha * d
        mean = self.base.unsqueeze(0) * scale
        return {"mean": mean}
