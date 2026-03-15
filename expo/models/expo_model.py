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
    """
    ExPO: Exposure-Response Neural Operator for Gene Expression Imputation
    
    ExPO is an exposure-conditioned neural operator that predicts full L1000 (978-gene) 
    z-score signatures for a given compound–cell context as a continuous function of 
    dose and time exposure conditions.
    
    Architecture:
    - DeepONet structure with branch and trunk networks
    - ChemBERTa-2 molecular embeddings with LoRA adaptation for chemical encoding
    - Sinusoidal Fourier features for continuous dose-time representation
    - Quantile heads for uncertainty quantification with conformal calibration
    
    The model enables evaluation at arbitrary dose–time pairs without regridding,
    supporting in-silico exploration of dose–time response surfaces.
    """

    def __init__(
        self,
        num_cells: int,
        exposure_feat_dim: int,
        cfg: ExperimentConfig,
        n_genes: int = DEFAULT_NUM_GENES,
    ) -> None:
        """
        Initialize ExPO model.
        
        Args:
            num_cells: Number of unique cell lines in dataset
            exposure_feat_dim: Dimensionality of exposure features (dose-time)
            cfg: Experiment configuration containing model hyperparameters
            n_genes: Number of genes to predict (default: 978 for L1000)
        """
        super().__init__()
        self.cfg = cfg
        self.n_genes = n_genes

        # Chemical encoder: ChemBERTa-2 with LoRA adaptation
        # Processes SMILES strings into molecular embeddings
        self.chem_encoder = ChemBERTaEncoder(
            model_name=cfg.chem.pretrained_model_name,
            max_length=cfg.chem.max_smiles_length,
            freeze_lower_layers=cfg.chem.freeze_lower_layers,
            lora_rank=cfg.chem.lora_rank,  # LoRA rank for efficient adaptation
            lora_alpha=cfg.chem.lora_alpha,  # LoRA scaling parameter
            lora_dropout=cfg.chem.lora_dropout,  # LoRA dropout rate
        )
        
        # Cell context encoder: handles cell line information and basal expression
        self.cell_context = CellContextEncoder(
            num_cells=num_cells,
            dim=cfg.cell.embedding_dim,
            use_basal=cfg.cell.use_basal_expression,
            basal_dim=n_genes,
            dropout=cfg.cell.basal_projection_dropout,
        )
        
        # Context fusion: combines chemical and cellular contexts
        self.context_fusion = ContextFusionMLP(
            chem_dim=self.chem_encoder.out_dim,
            cell_dim=cfg.cell.embedding_dim,
            out_dim=cfg.operator.context_dim,
        )

        # DeepONet branch network: processes fused context (compound + cell)
        # Maps context to basis coefficients for the neural operator
        self.branch = DeepONetBranch(
            in_dim=cfg.operator.context_dim,
            r=cfg.operator.r,  # Rank of the neural operator decomposition
            hidden=cfg.operator.branch_hidden,
        )
        
        # DeepONet trunk network: processes exposure features (dose-time)
        # Maps sinusoidal Fourier features to basis functions
        self.trunk = DeepONetTrunk(
            in_dim=exposure_feat_dim,  # Sinusoidal Fourier features dimension
            r=cfg.operator.r,  # Matching rank for tensor product
            hidden=cfg.operator.trunk_hidden,
        )
        
        # Mean prediction head: combines branch coefficients and trunk basis
        self.head_mean = DeepONetHead(cfg.operator.r, out_dim=n_genes)

        # Quantile heads for uncertainty quantification
        # Enables conformal prediction and calibrated confidence intervals
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
        """
        Encode compound-cell context information.
        
        Combines ChemBERTa molecular embeddings with cell line embeddings
        and optional basal expression profiles.
        
        Args:
            smiles: List of SMILES strings for compounds
            cell_indices: Tensor of cell line indices
            basal_expression: Optional basal gene expression profiles
            device: Compute device (CPU/GPU)
            
        Returns:
            Fused context embeddings
        """
        # Encode molecular structure using ChemBERTa-2 with LoRA
        chem_emb = self.chem_encoder(smiles, device=device)
        
        # Encode cell context (cell line + optional basal expression)
        cell_emb = self.cell_context(
            cell_indices.to(device),
            basal_expression=basal_expression.to(device) if basal_expression is not None else None,
        )
        
        # Fuse chemical and cellular contexts
        return self.context_fusion(chem_emb, cell_emb)

    def forward(  # type: ignore[override]
        self,
        smiles: Sequence[str],
        cell_indices: torch.Tensor,
        exposure_feats: torch.Tensor,
        basal_expression: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ExPO model.
        
        Predicts gene expression signatures as a continuous function of 
        dose-time exposure conditions using the DeepONet architecture.
        
        Args:
            smiles: SMILES strings for compounds
            cell_indices: Cell line indices
            exposure_feats: Sinusoidal Fourier features encoding dose-time
            basal_expression: Optional basal expression profiles
            device: Compute device
            
        Returns:
            Dictionary containing:
            - 'mean': Mean gene expression predictions (batch_size, n_genes)
            - 'quantiles': Quantile predictions if enabled (batch_size, n_quantiles, n_genes)
        """
        device = device or next(self.parameters()).device
        exposure_feats = exposure_feats.to(device)

        # Encode compound-cell context
        context = self.encode_context(smiles, cell_indices, basal_expression, device=device)
        
        # DeepONet forward pass:
        # Branch network maps context to basis coefficients
        branch_coef = self.branch(context)
        
        # Trunk network maps exposure features to basis functions
        trunk_basis = self.trunk(exposure_feats)
        
        # Combine via tensor product for mean prediction
        mean = self.head_mean(branch_coef, trunk_basis)

        out: Dict[str, torch.Tensor] = {"mean": mean}
        
        # Add quantile predictions for uncertainty quantification
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
    """
    Build ExPO model from configuration.
    
    Constructs the complete ExPO architecture with sinusoidal Fourier 
    features for dose-time encoding based on the provided exposure ranges.
    
    Args:
        cfg: Experiment configuration
        num_cells: Number of cell lines
        exposure_times: Range of exposure times for Fourier feature scaling
        exposure_doses: Range of exposure doses for Fourier feature scaling
        n_genes: Number of genes (default: 978 for L1000)
        
    Returns:
        Configured ExPO model instance
    """
    import numpy as np

    # Convert exposure ranges to numpy arrays
    times = np.asarray(exposure_times, dtype=np.float32)
    doses = np.asarray(exposure_doses, dtype=np.float32)
    
    # Build sinusoidal Fourier features for continuous dose-time encoding
    # This enables evaluation at arbitrary dose-time pairs without regridding
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
