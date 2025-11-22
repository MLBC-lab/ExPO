from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to a batch of logits."""
    t = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
    return logits / t.clamp_min(1e-6)


@dataclass
class TemperatureScaler(nn.Module):
    """Standard temperature scaling for classification logits.

    This module can be used to calibrate probabilities derived from logits
    (e.g., for discretized regulation direction) using a held-out validation
    set and negative log-likelihood as the objective.
    """

    init_temperature: float = 1.0

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(self.init_temperature))))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        t = self.log_temperature.exp().clamp_min(1e-6)
        return logits / t

    @torch.no_grad()
    def get_temperature(self) -> float:
        return float(self.log_temperature.exp().item())

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> "TemperatureScaler":
        """Optimize temperature on a validation set.

        Parameters
        ----------
        logits:
            Tensor of shape (n_samples, n_classes).
        labels:
            Integer class labels of shape (n_samples,).
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)

        labels = labels.long()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self
