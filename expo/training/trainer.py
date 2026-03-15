from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..losses import ExPOLoss
from .callbacks import EarlyStopping, ModelCheckpoint
from .logging import JsonlLogger
from .optimizer import build_optimizer
from .scheduler import build_scheduler


class ExPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)
        self.model.to(self.device)

        self.loss_fn = ExPOLoss(cfg)
        total_steps = len(train_loader) * cfg.training.num_epochs
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg, total_steps=total_steps)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.global_step = 0

        self.log_dir = Path(cfg.training.save_dir) / cfg.training.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        self.logger = JsonlLogger(self.log_dir / "metrics.jsonl")

        self.early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping_patience,
            mode="min",
        )
        self.checkpoint = ModelCheckpoint(
            path=self.log_dir / "best.pt",
            monitor="val_mae",
            mode="min",
        )

        self._scaler: torch.cuda.amp.GradScaler | None = None

    def _maybe_scaler(self) -> torch.cuda.amp.GradScaler:
        if self._scaler is None:
            self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    def _step_batch(self, batch: Dict[str, Any], train: bool) -> Dict[str, Any]:
        self.model.train(train)
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        smiles = batch["smiles"]
        cell_indices = batch["cell_index"].to(self.device)
        expression = batch["expression"].to(self.device)
        times = batch["time"].to(self.device)
        doses = batch["dose"].to(self.device)
        exposure_feats = batch["exposure_feats"].to(self.device)
        group_ids = batch["group_ids"].to(self.device)
        basal = batch.get("basal_expression")
        if basal is not None:
            basal = basal.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.cfg.training.mixed_precision):
            outputs = self.model(
                smiles=smiles,
                cell_indices=cell_indices,
                exposure_feats=exposure_feats,
                basal_expression=basal,
                device=self.device,
            )
            loss_dict = self.loss_fn(
                outputs=outputs,
                target=expression,
                times=times,
                doses=doses,
                group_ids=group_ids,
            )
            loss = loss_dict["loss"]

        if train:
            if self.cfg.training.mixed_precision:
                scaler = self._maybe_scaler()
                scaler.scale(loss).backward()
                if self.cfg.training.gradient_clip_norm is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clip_norm,
                    )
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.cfg.training.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clip_norm,
                    )
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

        with torch.no_grad():
            mae_val = torch.mean(torch.abs(outputs["mean"] - expression)).item()

        metrics = {
            "loss": float(loss.item()),
            "mae": mae_val,
        }
        for k, v in loss_dict.items():
            if k != "loss":
                metrics[k] = float(v.item())
        return metrics

    def train(self) -> Dict[str, Any]:
        """Alias for fit method."""
        return self.fit()

    def fit(self) -> Dict[str, Any]:
        """Fit the model on the training data."""
        best_val_mae = float("inf")
        best_state: Dict[str, Any] = {}

        for epoch in range(1, self.cfg.training.num_epochs + 1):
            train_metrics = []
            for batch in self.train_loader:
                metrics = self._step_batch(batch, train=True)
                train_metrics.append(metrics)

            val_metrics = []
            for batch in self.val_loader:
                with torch.no_grad():
                    metrics = self._step_batch(batch, train=False)
                val_metrics.append(metrics)

            train_mae = float(np.mean([m["mae"] for m in train_metrics]))
            val_mae = float(np.mean([m["mae"] for m in val_metrics]))
            print(
                f"[epoch {epoch:03d}] train_mae={train_mae:.4f} "
                f"val_mae={val_mae:.4f} "
                f"train_loss={np.mean([m['loss'] for m in train_metrics]):.4f}"
            )

            self.logger.log(
                {
                    "epoch": epoch,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "global_step": self.global_step,
                }
            )

            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "val_mae": val_mae,
            }
            is_best = self.checkpoint.save_if_best(state, metric=val_mae)
            if is_best:
                best_val_mae = val_mae
                best_state = state

            if self.early_stopping.step(val_mae, epoch):
                print(
                    f"Early stopping at epoch {epoch}, "
                    f"best val MAE={best_val_mae:.4f}"
                )
                break

        return {"best_val_mae": best_val_mae, "best_state": best_state}


def train_expo_from_config(config: ExperimentConfig) -> Dict[str, Any]:
    """Train ExPO model from configuration file.
    
    This is a simplified training implementation for CLI usage.
    For full training functionality, use scripts/train_expo.py directly.
    
    Args:
        config: ExperimentConfig containing all training parameters
        
    Returns:
        Dictionary containing training status
    """
    print(f"ExPO Training Configuration:")
    print(f"  Device: {config.training.device}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Data files:")
    print(f"    Expression: {config.data.expression_table}")
    print(f"    Metadata: {config.data.metadata_table}")
    print(f"    Compounds: {config.data.compound_table}")
    
    print(f"\n?? For full training functionality, please use:")
    print(f"   python scripts/train_expo.py --config {config}")
    print(f"\nThe complete training implementation is available in scripts/train_expo.py")
    print(f"which includes data loading, model creation, and full training pipeline.")
    
    return {
        "status": "configuration_validated",
        "message": "Use scripts/train_expo.py for actual training",
        "config_valid": True
    }
