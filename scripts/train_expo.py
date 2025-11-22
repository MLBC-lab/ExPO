import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from expo.config import ExperimentConfig
from expo.data import (
    CMapL1000Dataset,
    InMemoryDatasetCache,
    build_exposure_features,
    random_split_indices,
    scaffold_split_indices,
)
from expo.training import ExPOTrainer, set_global_seed
from expo.models import ExPOModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--fold", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.load(args.config)
    set_global_seed(cfg.training.seed)

    # Load tables
    expr_df = pd.read_pickle(cfg.data.expression_table)
    meta_df = pd.read_pickle(cfg.data.metadata_table)
    cmp_df = pd.read_pickle(cfg.data.compound_table)

    frame = meta_df.merge(expr_df, on="profile_id", how="inner")
    frame = frame.merge(cmp_df[["compound_id", "smiles"]], on="compound_id", how="left")

    # Build exposure features
    times = frame["time"].to_numpy(dtype=np.float32)
    doses = frame["dose"].to_numpy(dtype=np.float32)
    exp_feats = build_exposure_features(
        times=times,
        doses=doses,
        num_frequencies=cfg.exposure.fourier_frequencies,
        log_eps=cfg.exposure.log_eps,
        include_raw=cfg.exposure.include_raw_time_dose,
    )
    frame["exposure_feats"] = list(exp_feats)

    # Group ids
    group_ids = (
        frame["compound_id"].astype(str)
        + "|"
        + frame["cell_id"].astype(str)
        + "|"
        + frame["time"].astype(str)
    )
    frame["group_id"] = group_ids.apply(lambda s: hash(s) & 0xFFFFFFFF)

    cache = InMemoryDatasetCache.from_frame(frame)

    # Folds
    if cfg.data.split_scheme == "scaffold":
        folds = scaffold_split_indices(frame, cmp_df, cfg.data.n_folds, seed=cfg.data.split_seed)
    else:
        folds = random_split_indices(frame, cfg.data.n_folds, seed=cfg.data.split_seed)

    train_idx, val_idx = folds[args.fold]
    train_mask = np.zeros(len(frame), dtype=bool)
    val_mask = np.zeros(len(frame), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    train_ds = cache.materialize(
        mask=train_mask,
        randomized_smiles=cfg.chem.train_randomized_smiles,
        randomized_smiles_prob=cfg.chem.randomized_smiles_prob,
    )
    val_ds = cache.materialize(
        mask=val_mask,
        randomized_smiles=False,
        randomized_smiles_prob=0.0,
    )

    def collate(batch):
        smiles = [b["smiles"] for b in batch]
        cell_index = torch.tensor([b["cell_index"] for b in batch], dtype=torch.long)
        expr = torch.stack([torch.from_numpy(b["expression"]) for b in batch], dim=0)
        time = torch.tensor([b["time"] for b in batch], dtype=torch.float32)
        dose = torch.tensor([b["dose"] for b in batch], dtype=torch.float32)
        exposure_feats = torch.stack(
            [torch.from_numpy(b["exposure_feats"]) for b in batch],
            dim=0,
        )
        group_ids = torch.tensor([b["group_id"] for b in batch], dtype=torch.long)
        out = {
            "smiles": smiles,
            "cell_index": cell_index,
            "expression": expr,
            "time": time,
            "dose": dose,
            "exposure_feats": exposure_feats,
            "group_ids": group_ids,
        }
        if "basal_expression" in batch[0]:
            out["basal_expression"] = torch.stack(
                [torch.from_numpy(b["basal_expression"]) for b in batch],
                dim=0,
            )
        return out

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    num_cells = len(cache.cell_id_to_index)
    exposure_feat_dim = exp_feats.shape[1]
    model = ExPOModel(
        num_cells=num_cells,
        exposure_feat_dim=exposure_feat_dim,
        cfg=cfg,
        n_genes=expr_df.shape[1],
    )

    trainer = ExPOTrainer(model=model, cfg=cfg, train_loader=train_loader, val_loader=val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
