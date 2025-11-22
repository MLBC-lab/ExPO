from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_risk_coverage_curve(
    coverages: np.ndarray,
    risks: np.ndarray,
    title: str = "Risk–coverage curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot risk vs coverage (lower is better)."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(coverages, risks, marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (e.g., MAE)")
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_rank_metric_bars(
    metrics: Dict[str, float],
    title: str = "Ranking metrics",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Bar chart of several ranking metrics (NDCG, Jaccard, RBO, etc.)."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    names = list(metrics.keys())
    vals = [float(metrics[k]) for k in names]
    x = np.arange(len(names))
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    fig.tight_layout()
    return ax
