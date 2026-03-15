from __future__ import annotations

import numpy as np
from typing import Tuple


def build_exposure_features(
    times: np.ndarray,
    doses: np.ndarray,
    num_frequencies: int = 16,
    log_eps: float = 1e-3,
    include_raw: bool = True,
) -> np.ndarray:
    """
    Build sinusoidal Fourier features for continuous dose-time encoding.
    
    As described in the ExPO paper, this function creates sinusoidal Fourier 
    features that enable the trunk network to process dose-time pairs as 
    continuous functions rather than discrete buckets. This allows evaluation 
    at arbitrary dose-time coordinates without regridding.
    
    The Fourier features capture multi-scale periodic patterns in the 
    dose-time space, providing the trunk network with rich representations
    for modeling complex exposure-response relationships.
    
    Args:
        times: Array of exposure times (hours)
        doses: Array of exposure doses (concentration units, e.g., ?M)
        num_frequencies: Number of sinusoidal frequency components
        log_eps: Small epsilon for log transformation to handle zero doses
        include_raw: Whether to include raw time/dose values alongside Fourier features
        
    Returns:
        Array of sinusoidal Fourier features with shape (n_samples, n_features)
        Feature dimension = 4 * num_frequencies + (2 if include_raw else 0)
    """
    n_samples = len(times)
    assert len(doses) == n_samples, "Times and doses must have same length"
    
    features = []
    
    # Log-transform doses to handle wide dynamic range
    # Adding epsilon prevents log(0) for zero-dose controls
    log_doses = np.log(doses + log_eps)
    
    # Normalize to [0, 1] range for stable Fourier feature computation
    # This ensures consistent scaling across different experimental ranges
    time_min, time_max = times.min(), times.max()
    dose_min, dose_max = log_doses.min(), log_doses.max()
    
    # Handle edge case where all values are identical
    time_range = time_max - time_min
    dose_range = dose_max - dose_min
    
    if time_range > 1e-8:
        time_norm = (times - time_min) / time_range
    else:
        time_norm = np.zeros_like(times)
        
    if dose_range > 1e-8:
        dose_norm = (log_doses - dose_min) / dose_range
    else:
        dose_norm = np.zeros_like(log_doses)
    
    # Generate sinusoidal Fourier features across multiple frequencies
    # Each frequency captures different scales of periodicity in exposure space
    for i in range(1, num_frequencies + 1):
        freq = 2.0 * np.pi * i
        
        # Time-based sinusoidal features
        # Capture temporal dynamics at different scales
        features.append(np.sin(freq * time_norm))
        features.append(np.cos(freq * time_norm))
        
        # Dose-based sinusoidal features  
        # Capture dose-response relationships at different scales
        features.append(np.sin(freq * dose_norm))
        features.append(np.cos(freq * dose_norm))
    
    # Optionally include raw time and log-dose values
    # Provides direct access to exposure coordinates alongside Fourier encoding
    if include_raw:
        features.append(times)
        features.append(log_doses)
    
    # Stack all features into final array
    return np.column_stack(features)


def get_exposure_feature_dim(num_frequencies: int = 16, include_raw: bool = True) -> int:
    """
    Calculate the dimensionality of exposure features.
    
    Args:
        num_frequencies: Number of Fourier frequency components
        include_raw: Whether raw time/dose are included
        
    Returns:
        Feature dimensionality: 4 * num_frequencies + (2 if include_raw else 0)
    """
    fourier_dim = 4 * num_frequencies  # sin/cos for both time and dose
    raw_dim = 2 if include_raw else 0  # raw time and log-dose
    return fourier_dim + raw_dim
