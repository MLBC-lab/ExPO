from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import ExperimentConfig
from .quantile_loss import multi_quantile_loss
from .ranking import two_list_listnet_loss
from .regression import huber_loss
from .regularization import sobolev_penalty, dose_monotonicity_penalty


class ExPOLoss(nn.Module):
    """Composite training loss used for ExPO."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        times: torch.Tensor,
        doses: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        mean_pred = outputs["mean"]

        loss_reg = huber_loss(
            mean_pred,
            target,
            delta=self.cfg.loss.huber_delta,
        ).mean()

        loss_list = two_list_listnet_loss(
            pred=mean_pred,
            target=target,
            up_thresh=self.cfg.data.up_threshold,
            down_thresh=self.cfg.data.down_threshold,
            temperature=self.cfg.loss.listnet_temperature,
        )

        loss_sob = sobolev_penalty(
            model=None,
            mean_pred=mean_pred,
            times=times,
            doses=doses,
            lambda_first=self.cfg.loss.sobolev_first,
            lambda_second=self.cfg.loss.sobolev_second,
        )

        loss_mono = dose_monotonicity_penalty(
            mean_pred=mean_pred,
            doses=doses,
            group_ids=group_ids,
            margin=0.0,
        )

        if "quantiles" in outputs and self.cfg.quantile.enabled:
            q_loss = multi_quantile_loss(
                outputs["quantiles"],
                target,
                quantiles=self.cfg.quantile.quantiles,
            )
        else:
            q_loss = mean_pred.new_tensor(0.0)

        total = (
            self.cfg.loss.lambda_regression * loss_reg
            + self.cfg.loss.lambda_listnet * loss_list
            + self.cfg.loss.lambda_sobolev * loss_sob
            + self.cfg.loss.lambda_monotonicity * loss_mono
            + self.cfg.loss.lambda_quantile * q_loss
        )

        return {
            "loss": total,
            "loss_reg": loss_reg.detach(),
            "loss_list": loss_list.detach(),
            "loss_sob": loss_sob.detach(),
            "loss_mono": loss_mono.detach(),
            "loss_quantile": q_loss.detach(),
        }
