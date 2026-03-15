from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import randomize_smiles


class CMapL1000Dataset(Dataset):
    """Dataset exposing one row per (compound, cell, time, dose) profile."""

    def __init__(
        self,
        frame: pd.DataFrame,
        cell_id_to_index: Dict[Any, int],
        randomized_smiles: bool,
        randomized_smiles_prob: float,
    ) -> None:
        super().__init__()
        self._frame = frame.reset_index(drop=True)
        self._cell_id_to_index = dict(cell_id_to_index)
        self._randomized_smiles = randomized_smiles
        self._randomized_smiles_prob = randomized_smiles_prob

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._frame)

    def _maybe_randomize(self, smiles: str) -> str:
        import random

        if self._randomized_smiles and random.random() < self._randomized_smiles_prob:
            return randomize_smiles(smiles)
        return smiles

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        row = self._frame.iloc[idx]
        smiles = self._maybe_randomize(str(row["smiles"]))
        cell_idx = self._cell_id_to_index[row["cell_id"]]
        expr = np.asarray(row["expression"], dtype=np.float32)
        exposure_feats = np.asarray(row["exposure_feats"], dtype=np.float32)

        sample: Dict[str, Any] = {
            "compound_id": row["compound_id"],
            "cell_id": row["cell_id"],
            "cell_index": cell_idx,
            "smiles": smiles,
            "time": float(row["time"]),
            "dose": float(row["dose"]),
            "expression": expr,
            "exposure_feats": exposure_feats,
            "group_id": int(row["group_id"]),
        }
        if "basal_expression" in row:
            sample["basal_expression"] = np.asarray(row["basal_expression"], dtype=np.float32)
        return sample


# Alias for backward compatibility and API consistency
ExPODataset = CMapL1000Dataset


@dataclass
class InMemoryDatasetCache:
    """Simple in-memory cache for datasets.

    This is intentionally light-weight: it stores numpy arrays detached from
    Pandas and can rebuild a new dataset view when needed.
    """

    frame: pd.DataFrame
    cell_id_to_index: Dict[Any, int]

    def materialize(
        self,
        mask: Optional[np.ndarray] = None,
        randomized_smiles: bool = False,
        randomized_smiles_prob: float = 0.0,
    ) -> CMapL1000Dataset:
        if mask is not None:
            sub = self.frame[mask].reset_index(drop=True)
        else:
            sub = self.frame
        return CMapL1000Dataset(
            frame=sub,
            cell_id_to_index=self.cell_id_to_index,
            randomized_smiles=randomized_smiles,
            randomized_smiles_prob=randomized_smiles_prob,
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "InMemoryDatasetCache":
        cells = sorted(frame["cell_id"].unique().tolist())
        mapping = {c: i for i, c in enumerate(cells)}
        return cls(frame=frame.reset_index(drop=True), cell_id_to_index=mapping)
