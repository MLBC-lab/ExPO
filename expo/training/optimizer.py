from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW

from ..config import ExperimentConfig


def _iter_trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    for p in model.parameters():
        if p.requires_grad:
            yield p


def build_optimizer(model: torch.nn.Module, cfg: ExperimentConfig) -> AdamW:
    params = list(_iter_trainable_parameters(model))
    return AdamW(
        params,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
