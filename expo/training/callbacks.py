from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    best_score: Optional[float] = None
    num_bad_epochs: int = 0
    stopped_epoch: Optional[int] = None

    def step(self, metric: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False

        improve = metric < self.best_score if self.mode == "min" else metric > self.best_score
        if improve:
            self.best_score = metric
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stopped_epoch = epoch
                return True
            return False


@dataclass
class ModelCheckpoint:
    path: Path
    monitor: str = "val_mae"
    mode: str = "min"
    best_score: Optional[float] = None

    def save_if_best(self, state: Dict[str, Any], metric: float) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.best_score is None:
            self.best_score = metric
            torch.save(state, self.path)
            return True

        better = metric < self.best_score if self.mode == "min" else metric > self.best_score
        if better:
            self.best_score = metric
            torch.save(state, self.path)
            return True
        return False
