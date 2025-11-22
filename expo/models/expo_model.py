from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from ..constants import DEFAULT_NUM_GENES
from ..config import ExperimentConfig
from ..data.exposure_features import build_exposure_features
from .cell_encoder import CellContextEncoder
from .chem_encoder import ChemBERTaEncoder
from .context import ContextFusionMLP
from .deeponet import DeepONetBranch, DeepONetTrunk, DeepONetHead
from .quantile import QuantileHead


class ExPOModel(nn.Module):
    """Full ExPO model implementation."""

    def __init__(
        self,
        num_cells: int,
        exposure_feat_dim: int,
        cfg: ExperimentConfig,
        n_genes: int = DEFAULT_NUM_GENES,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_genes = n_genes

        self.chem_encoder = ChemBERTaEncoder(
            model_name=cfg.chem.pretrained_model_name,
            max_length=cfg.chem.max_smiles_length,
            freeze_lower_layers=cfg.chem.freeze_lower_layers,
            lora_rank=cfg.chem.lora_rank,
            lora_alpha=cfg.chem.lora_alpha,
            lora_dropout=cfg.chem.lora_dropout,
        )
        self.cell_context = CellContextEncoder(
            num_cells=num_cells,
            dim=cfg.cell.embedding_dim,
            use_basal=cfg.cell.use_basal_expression,
            basal_dim=n_genes,
            dropout=cfg.cell.basal_projection_dropout,
        )
        self.context_fusion = ContextFusionMLP(
            chem_dim=self.chem_encoder.out_dim,
            cell_dim=cfg.cell.embedding_dim,
            out_dim=cfg.operator.context_dim,
        )

        self.branch = DeepONetBranch(
            in_dim=cfg.operator.context_dim,
            r=cfg.operator.r,
            hidden=cfg.operator.branch_hidden,
        )
        self.trunk = DeepONetTrunk(
            in_dim=exposure_feat_dim,
            r=cfg.operator.r,
            hidden=cfg.operator.trunk_hidden,
        )
        self.head_mean = DeepONetHead(cfg.operator.r, out_dim=n_genes)

        self.quantile_head: Optional[QuantileHead]
        if cfg.quantile.enabled:
            self.quantile_head = QuantileHead(
                r=cfg.operator.r,
                out_dim=n_genes,
                n_quantiles=len(cfg.quantile.quantiles),
            )
        else:
            self.quantile_head = None

    def encode_context(
        self,
        smiles: Sequence[str],
        cell_indices: torch.Tensor,
        basal_expression: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        chem_emb = self.chem_encoder(smiles, device=device)
        cell_emb = self.cell_context(
            cell_indices.to(device),
            basal_expression=basal_expression.to(device) if basal_expression is not None else None,
        )
        return self.context_fusion(chem_emb, cell_emb)

    def forward(  # type: ignore[override]
        self,
        smiles: Sequence[str],
        cell_indices: torch.Tensor,
        exposure_feats: torch.Tensor,
        basal_expression: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        device = device or next(self.parameters()).device
        exposure_feats = exposure_feats.to(device)

        context = self.encode_context(smiles, cell_indices, basal_expression, device=device)
        branch_coef = self.branch(context)
        trunk_basis = self.trunk(exposure_feats)
        mean = self.head_mean(branch_coef, trunk_basis)

        out: Dict[str, torch.Tensor] = {"mean": mean}
        if self.quantile_head is not None:
            q = self.quantile_head(branch_coef, trunk_basis)
            out["quantiles"] = q
        return out


def build_expo_from_config(
    cfg: ExperimentConfig,
    num_cells: int,
    exposure_times: Sequence[float],
    exposure_doses: Sequence[float],
    n_genes: int = DEFAULT_NUM_GENES,
) -> ExPOModel:
    import numpy as np

    times = np.asarray(exposure_times, dtype=np.float32)
    doses = np.asarray(exposure_doses, dtype=np.float32)
    feats = build_exposure_features(
        times=times,
        doses=doses,
        num_frequencies=cfg.exposure.fourier_frequencies,
        log_eps=cfg.exposure.log_eps,
        include_raw=cfg.exposure.include_raw_time_dose,
    )
    exposure_feat_dim = feats.shape[1]
    return ExPOModel(
        num_cells=num_cells,
        exposure_feat_dim=exposure_feat_dim,
        cfg=cfg,
        n_genes=n_genes,
    )
