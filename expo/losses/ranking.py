from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def listnet_distribution(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute ListNet probability distribution from scores.
    
    Applies temperature scaling and softmax to convert raw scores 
    into probability distributions for ranking objectives.
    
    Args:
        scores: Raw prediction scores
        temperature: Temperature parameter for softmax scaling
        
    Returns:
        Probability distribution over rankings
    """
    scaled = scores / temperature
    return F.softmax(scaled, dim=-1)


def two_list_listnet_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    up_thresh: float,
    down_thresh: float,
    temperature: float,
) -> torch.Tensor:
    """
    Two-list ListNet ranking objective for gene expression prediction.
    
    As described in the ExPO paper, this implements a "two-list listwise objective
    to optimize early-rank gene ordering" by separately optimizing rankings for
    up-regulated and down-regulated gene sets. This is discovery-aligned training
    that focuses on correctly ranking the most responsive genes, which are most
    important for biological interpretation and drug discovery applications.
    
    The loss computes separate ListNet objectives for:
    1. Up-regulated genes (expression >= up_thresh)
    2. Down-regulated genes (expression <= down_thresh)
    
    This approach ensures that the model learns to properly rank genes within
    each response category, improving both accuracy and biological relevance
    of the predicted gene expression signatures.
    
    Args:
        pred: Predicted gene expression scores (batch_size, n_genes)
        target: Target gene expression z-scores (batch_size, n_genes)
        up_thresh: Threshold for up-regulated genes (e.g., 1.0 z-score)
        down_thresh: Threshold for down-regulated genes (e.g., -1.0 z-score)
        temperature: Temperature parameter for ListNet softmax scaling
        
    Returns:
        Two-list ListNet loss combining up- and down-regulated gene rankings
    """
    B, G = target.shape
    device = target.device

    # Create masks for up- and down-regulated gene sets
    # These define the two lists for the listwise ranking objective
    up_mask = target >= up_thresh      # Up-regulated genes
    down_mask = target <= down_thresh  # Down-regulated genes

    loss_terms = []

    # Compute ListNet loss separately for each gene set
    for mask in (up_mask, down_mask):
        if not mask.any():
            continue  # Skip if no genes in this category
            
        # Flatten and select genes in current category
        idx = mask.view(-1)
        if not idx.any():
            continue

        # Extract target and prediction scores for selected genes
        t = target.view(-1)[idx].view(B, -1)  # Target z-scores
        p = pred.view(-1)[idx].view(B, -1)    # Predicted scores

        # Convert to probability distributions using ListNet
        # Target distribution based on absolute z-score magnitudes
        t_dist = listnet_distribution(t.abs(), temperature=temperature)
        # Prediction distribution based on model outputs
        p_dist = listnet_distribution(p, temperature=temperature)
        
        # Cross-entropy loss between target and predicted ranking distributions
        ce = -(t_dist * (p_dist + 1e-9).log()).sum(dim=-1)
        loss_terms.append(ce.mean())

    # Return combined loss or zero if no responsive genes found
    if not loss_terms:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return sum(loss_terms) / len(loss_terms)
