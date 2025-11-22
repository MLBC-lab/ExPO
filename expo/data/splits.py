from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .preprocessing import add_scaffold_column


def scaffold_split_indices(
    frame: pd.DataFrame,
    compounds_df: pd.DataFrame,
    n_folds: int,
    seed: int = 42,
    compound_col: str = "compound_id",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Assign compounds to folds based on Bemis–Murcko scaffold.

    Returns a list of (train_idx, val_idx) arrays covering *frame*.
    """
    rng = random.Random(seed)
    compounds_df = add_scaffold_column(compounds_df, smiles_col="smiles")
    merged = frame[[compound_col]].merge(
        compounds_df[[compound_col, "scaffold"]],
        on=compound_col,
        how="left",
    )
    scaffolds = merged["scaffold"].fillna("").values
    unique_scaffolds = sorted(set(scaffolds))
    rng.shuffle(unique_scaffolds)

    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, scaf in enumerate(unique_scaffolds):
        folds[i % n_folds].append(scaf)

    all_indices = np.arange(len(frame))
    per_fold: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        val_scaf = set(folds[i])
        val_mask = np.isin(scaffolds, list(val_scaf))
        val_idx = all_indices[val_mask]
        train_idx = all_indices[~val_mask]
        per_fold.append((train_idx, val_idx))
    return per_fold


def random_split_indices(frame: pd.DataFrame, n_folds: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)
    chunks = np.array_split(indices, n_folds)
    per_fold: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        val_idx = chunks[i]
        train_idx = np.concatenate([chunks[j] for j in range(n_folds) if j != i])
        per_fold.append((train_idx, val_idx))
    return per_fold


def leave_cell_line_out_splits(frame: pd.DataFrame, cell_col: str = "cell_id") -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-cell-line-out split indices.

    Each split returns (train_idx, val_idx) such that the validation indices
    correspond to a single held-out cell line.
    """
    all_indices = np.arange(len(frame))
    per_split: List[Tuple[np.ndarray, np.ndarray]] = []
    for cell in sorted(frame[cell_col].unique()):
        mask = frame[cell_col] == cell
        val_idx = all_indices[mask.values]
        train_idx = all_indices[~mask.values]
        per_split.append((train_idx, val_idx))
    return per_split
