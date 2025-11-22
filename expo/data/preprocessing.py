from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover - rdkit is optional
    Chem = None
    MurckoScaffold = None


@dataclass
class SmilesStandardizationReport:
    n_total: int
    n_failed: int
    n_unique: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "n_total": self.n_total,
            "n_failed": self.n_failed,
            "n_unique": self.n_unique,
        }


def _require_rdkit() -> None:
    if Chem is None or MurckoScaffold is None:
        raise ImportError(
            "RDKit is required for scaffold-based operations but is not installed. "
            "Install with `conda install -c conda-forge rdkit`."
        )


def standardize_smiles(smiles: str) -> str:
    """Canonicalize and lightly sanitize a SMILES string.

    If RDKit is not available or parsing fails, the input is returned as-is.
    """
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        frags = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)
        mol = frags[0]
    return Chem.MolToSmiles(mol, canonical=True)


def randomize_smiles(smiles: str, rng: Optional[random.Random] = None) -> str:
    if Chem is None:
        return smiles
    rng = rng or random
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    atoms = list(range(mol.GetNumAtoms()))
    rng.shuffle(atoms)
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True)


def bemis_murcko_scaffold(smiles: str) -> str:
    if Chem is None:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def add_scaffold_column(compounds_df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    _require_rdkit()
    scaffolds: List[str] = []
    for smi in compounds_df[smiles_col].astype(str):
        scaffolds.append(bemis_murcko_scaffold(smi))
    df = compounds_df.copy()
    df["scaffold"] = scaffolds
    return df


def standardize_smiles_column(df: pd.DataFrame, col: str = "smiles") -> SmilesStandardizationReport:
    """Apply standardization to an entire column and return a simple report."""
    std = []
    n_failed = 0
    for s in df[col].astype(str):
        out = standardize_smiles(s)
        if out == s:
            # Could be a failure or already canonical; we do not try to distinguish.
            pass
        if Chem is not None and Chem.MolFromSmiles(out) is None:
            n_failed += 1
        std.append(out)
    df[col] = std
    return SmilesStandardizationReport(
        n_total=len(std),
        n_failed=n_failed,
        n_unique=df[col].nunique(),
    )
