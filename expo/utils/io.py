from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .typing_utils import PathLike


def load_table(path: PathLike) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")
