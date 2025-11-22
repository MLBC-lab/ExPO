from __future__ import annotations

from typing import Dict

import numpy as np


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """Return simple summary statistics for an embedding matrix."""
    return {
        "dim": float(embeddings.shape[-1]),
        "n": float(embeddings.shape[0]),
        "mean_norm": float(np.linalg.norm(embeddings, axis=-1).mean()),
        "std_norm": float(np.linalg.norm(embeddings, axis=-1).std()),
    }
