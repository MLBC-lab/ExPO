from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover
    PCA = None  # type: ignore[assignment]
    TSNE = None  # type: ignore[assignment]


def _project_2d(embeddings: np.ndarray, method: str = "pca") -> np.ndarray:
    if method.lower() == "pca":
        if PCA is None:
            raise ImportError("scikit-learn is required for PCA projections.")
        pca = PCA(n_components=2)
        return pca.fit_transform(embeddings)
    elif method.lower() == "tsne":
        if TSNE is None:
            raise ImportError("scikit-learn is required for t-SNE projections.")
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
        return tsne.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown projection method: {method}")


def plot_embedding_2d(
    embeddings: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    method: str = "pca",
    title: str = "Embedding projection",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of 2D embeddings, optionally colored by label."""
    proj = _project_2d(embeddings, method=method)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if labels is None:
        ax.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.6)
    else:
        labels = np.asarray(labels)
        uniq = np.unique(labels)
        for lab in uniq:
            mask = labels == lab
            ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.6, label=str(lab))
        ax.legend(markerscale=2.0, fontsize=8)

    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title + f" ({method.upper()})")
    fig.tight_layout()
    return ax
