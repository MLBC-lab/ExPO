from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..config import ExperimentConfig


def build_scheduler(optimizer: Optimizer, cfg: ExperimentConfig, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < cfg.training.warmup_steps:
            return float(step + 1) / float(max(1, cfg.training.warmup_steps))
        return max(
            0.0,
            float(total_steps - step)
            / float(max(1, total_steps - cfg.training.warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)
