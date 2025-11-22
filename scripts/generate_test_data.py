from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import json

@dataclass
class DataGenerationConfig:
    """Configuration for synthetic gene-expression-like data generation.

    The goal of this module is to produce structured arrays that *resemble*
    typical perturbational transcriptomics tables in shape and metadata
    layout, while remaining fully synthetic and controllable for testing
    and demonstration purposes.
    """

    n_profiles: int = 512
    n_genes: int = 200
    n_cells: int = 8
    n_compounds: int = 48
    random_seed: int = 13
    time_grid: Tuple[float, ...] = (6.0, 24.0, 48.0)
    dose_grid: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    latent_rank: int = 6
    cell_effect_scale: float = 0.6
    compound_effect_scale: float = 0.9
    interaction_scale: float = 0.5
    noise_scale: float = 0.5

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _sample_cell_loadings(cfg: DataGenerationConfig, rng: np.random.Generator) -> np.ndarray:
    # Latent factors per cell line
    return rng.normal(0.0, cfg.cell_effect_scale, size=(cfg.n_cells, cfg.latent_rank))


def _sample_compound_loadings(cfg: DataGenerationConfig, rng: np.random.Generator) -> np.ndarray:
    # Latent factors per compound
    return rng.normal(0.0, cfg.compound_effect_scale, size=(cfg.n_compounds, cfg.latent_rank))


def _sample_gene_loadings(cfg: DataGenerationConfig, rng: np.random.Generator) -> np.ndarray:
    # Latent sensitivities per gene
    return rng.normal(0.0, 1.0, size=(cfg.latent_rank, cfg.n_genes))


def _simulate_profile_matrix(
    cfg: DataGenerationConfig,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate a block of profiles with structured dependencies.

    This is deliberately more involved than a simple i.i.d. Gaussian generator:
    - Each cell line has its own latent embedding.
    - Each compound has its own potency vector.
    - Time and dose enter through smooth, non-linear transformations.
    - Residual noise is added at the end.
    """
    # Prepare grids
    times = np.asarray(cfg.time_grid, dtype=float)
    doses = np.asarray(cfg.dose_grid, dtype=float)

    # Latent structures
    cell_load = _sample_cell_loadings(cfg, rng)
    cmp_load = _sample_compound_loadings(cfg, rng)
    gene_load = _sample_gene_loadings(cfg, rng)

    profiles: List[Dict[str, object]] = []
    expr_rows: List[np.ndarray] = []

    # Construct indexable lists for IDs
    cell_ids = [f"CELL_{i:02d}" for i in range(cfg.n_cells)]
    cmp_ids = [f"CMP_{i:03d}" for i in range(cfg.n_compounds)]
    gene_names = [f"G{i:04d}" for i in range(cfg.n_genes)]

    # Precompute smooth dose/time response curves
    def dose_effect(d: float) -> float:
        return float(np.log1p(d) / np.log1p(doses.max()))

    def time_effect(t: float) -> float:
        return float((1.0 - np.exp(-t / 12.0)))

    profile_id = 0
    # We will oversample from all combinations until reaching n_profiles
    while profile_id < cfg.n_profiles:
        cell_idx = rng.integers(0, cfg.n_cells)
        cmp_idx = rng.integers(0, cfg.n_compounds)
        t = float(rng.choice(times))
        d = float(rng.choice(doses))

        cell_vec = cell_load[cell_idx]  # (latent_rank,)
        cmp_vec = cmp_load[cmp_idx]     # (latent_rank,)

        # Basic latent interaction
        latent = (cell_vec + cmp_vec) * time_effect(t) * dose_effect(d)
        # Map to genes
        baseline = latent @ gene_load  # (n_genes,)

        # Add a simple non-linear saturation at high doses
        saturation = 1.0 / (1.0 + np.exp(-d + doses.mean()))
        profile_signal = baseline * saturation

        noise = rng.normal(0.0, cfg.noise_scale, size=cfg.n_genes)
        expr = profile_signal + noise

        expr_rows.append(expr.astype("float32"))
        profiles.append(
            {
                "profile_id": profile_id,
                "compound_id": cmp_ids[cmp_idx],
                "cell_id": cell_ids[cell_idx],
                "time": t,
                "dose": d,
            }
        )
        profile_id += 1

    expr_arr = np.stack(expr_rows, axis=0)
    expr_df = pd.DataFrame(expr_arr, columns=gene_names)
    expr_df.insert(0, "profile_id", np.arange(cfg.n_profiles, dtype=int))

    meta_df = pd.DataFrame(profiles)

    # Construct a compound table with simple but valid-ish SMILES scaffolds
    cmp_df = _make_compound_table(cmp_ids, rng)

    return expr_df, meta_df, cmp_df


def _make_compound_table(compound_ids: List[str], rng: np.random.Generator) -> pd.DataFrame:
    """Construct a table of compound identifiers and toy SMILES strings."""
    scaffolds = ["C", "CC", "CCC", "c1ccccc1", "CO", "CN", "OC(=O)C"]
    smiles_list: List[str] = []
    for cid in compound_ids:
        base = scaffolds[rng.integers(0, len(scaffolds))]
        # Randomly append a small fragment to make surface-level variety
        if rng.random() < 0.5:
            frag = rng.choice(["F", "Cl", "Br", "N", "O"])
            smiles_list.append(base + frag)
        else:
            smiles_list.append(base)
    return pd.DataFrame(
        {
            "compound_id": compound_ids,
            "smiles": smiles_list,
        }
    )


def generate_dataset(out_dir: Path, cfg: DataGenerationConfig) -> Dict[str, Path]:
    """Generate all synthetic tables and write them to *out_dir* as pickles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = _make_rng(cfg.random_seed)

    expr_df, meta_df, cmp_df = _simulate_profile_matrix(cfg, rng)

    expr_path = out_dir / "expression_table.pkl"
    meta_path = out_dir / "metadata_table.pkl"
    cmp_path = out_dir / "compound_table.pkl"

    expr_df.to_pickle(expr_path)
    meta_df.to_pickle(meta_path)
    cmp_df.to_pickle(cmp_path)

    # Also save a compact JSON description of the generation parameters
    (out_dir / "generation_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2),
        encoding="utf-8",
    )

    return {
        "expression_table": expr_path,
        "metadata_table": meta_path,
        "compound_table": cmp_path,
    }


def parse_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="data/test")
    ap.add_argument("--n-profiles", type=int, default=512)
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--n-cells", type=int, default=8)
    ap.add_argument("--n-compounds", type=int, default=48)
    ap.add_argument("--seed", type=int, default=13)
    return ap


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    cfg = DataGenerationConfig(
        n_profiles=args.n_profiles,
        n_genes=args.n_genes,
        n_cells=args.n_cells,
        n_compounds=args.n_compounds,
        random_seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    table_paths = generate_dataset(out_dir, cfg)
    print("Synthetic test data written to:")
    for name, path in table_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
