from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _prepare_axes(ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def plot_dose_time_map(
    values: np.ndarray,
    times: np.ndarray,
    doses: np.ndarray,
    title: str = "Dose–time map",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Heatmap utility for dose–time surfaces."""
    fig, ax = _prepare_axes(ax)
    T, D = np.meshgrid(times, doses, indexing="ij")
    im = ax.pcolormesh(D, T, values, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Dose")
    ax.set_ylabel("Time (h)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return ax


def plot_profile_heatmap(
    expression: np.ndarray,
    gene_names: Sequence[str],
    title: str = "Expression profile",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a single profile as a heatmap across genes."""
    fig, ax = _prepare_axes(ax)
    expr = expression.reshape(1, -1)
    im = ax.imshow(expr, aspect="auto")
    ax.set_yticks([0])
    ax.set_yticklabels(["profile"])
    ax.set_xticks(np.arange(len(gene_names)))
    ax.set_xticklabels(gene_names, rotation=90, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return ax


def plot_metric_curves(
    history: Mapping[str, Sequence[float]],
    x: Optional[Sequence[float]] = None,
    title: str = "Training curves",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot multiple metric curves on the same axes."""
    fig, ax = _prepare_axes(ax)
    if x is None:
        n = len(next(iter(history.values())))
        x = list(range(1, n + 1))
    for name, vals in history.items():
        ax.plot(x, list(vals), label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return ax


def plot_error_histogram(
    errors: np.ndarray,
    bins: int = 50,
    title: str = "Absolute error histogram",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Histogram of absolute errors."""
    fig, ax = _prepare_axes(ax)
    ax.hist(errors, bins=bins)
    ax.set_xlabel("|error|")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_scatter_actual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs. Actual",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of predictions vs. ground truth with y=x line."""
    fig, ax = _prepare_axes(ax)
    ax.scatter(y_true, y_pred, s=4, alpha=0.5)
    lims = [
        min(float(np.min(y_true)), float(np.min(y_pred))),
        max(float(np.max(y_true)), float(np.max(y_pred))),
    ]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_violin_gene_distribution(
    expr_matrix: np.ndarray,
    gene_names: Sequence[str],
    max_genes: int = 50,
    title: str = "Gene expression distributions",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Violin plots for per-gene expression across profiles."""
    fig, ax = _prepare_axes(ax)
    n_genes = expr_matrix.shape[1]
    idx = np.arange(n_genes)
    if n_genes > max_genes:
        step = max(1, n_genes // max_genes)
        idx = idx[::step]
    data = [expr_matrix[:, i] for i in idx]
    ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    ax.set_xticks(np.arange(1, len(idx) + 1))
    ax.set_xticklabels([gene_names[i] for i in idx], rotation=90, fontsize=6)
    ax.set_title(title)
    ax.set_ylabel("Expression")
    fig.tight_layout()
    return ax


def plot_boxplot_metric_by_group(
    values: Sequence[float],
    groups: Sequence[str],
    title: str = "Metric by group",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Boxplot of metric values split by group label."""
    fig, ax = _prepare_axes(ax)
    values = np.asarray(values)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    data = [values[groups == g] for g in uniq]
    ax.boxplot(data, labels=uniq)
    ax.set_title(title)
    ax.set_ylabel("Metric value")
    fig.tight_layout()
    return ax
