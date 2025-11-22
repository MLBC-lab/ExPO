import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from expo.config import ExperimentConfig
from expo.data import InMemoryDatasetCache
from expo.metrics import mae, rank_metrics_for_profile
from expo.models import ExPOModel
from expo.utils.serialization import load_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = ExperimentConfig.load(args.config)

    expr_df = pd.read_pickle(cfg.data.expression_table)
    meta_df = pd.read_pickle(cfg.data.metadata_table)
    cmp_df = pd.read_pickle(cfg.data.compound_table)

    frame = meta_df.merge(expr_df, on="profile_id", how="inner")
    frame = frame.merge(cmp_df[["compound_id", "smiles"]], on="compound_id", how="left")

    cache = InMemoryDatasetCache.from_frame(frame)

    def collate(batch):
        smiles = [b["smiles"] for b in batch]
        cell_index = torch.tensor([b["cell_index"] for b in batch], dtype=torch.long)
        expr = torch.stack([torch.from_numpy(b["expression"]) for b in batch], dim=0)
        exposure_feats = torch.stack(
            [torch.from_numpy(b["exposure_feats"]) for b in batch],
            dim=0,
        )
        return {
            "smiles": smiles,
            "cell_index": cell_index,
            "expression": expr,
            "exposure_feats": exposure_feats,
        }

    loader = DataLoader(
        cache.materialize(),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=collate,
    )

    device = torch.device(cfg.training.device)
    num_cells = len(cache.cell_id_to_index)
    exposure_feat_dim = frame.iloc[0]["exposure_feats"].shape[0]
    model = ExPOModel(
        num_cells=num_cells,
        exposure_feat_dim=exposure_feat_dim,
        cfg=cfg,
        n_genes=expr_df.shape[1],
    )
    model, _ = load_model(args.checkpoint, model, map_location=str(device))
    model.to(device)
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            out = model(
                smiles=batch["smiles"],
                cell_indices=batch["cell_index"].to(device),
                exposure_feats=batch["exposure_feats"].to(device),
                device=device,
            )
            preds.append(out["mean"].cpu().numpy())
            trues.append(batch["expression"].numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    print("MAE:", mae(preds, trues))

    # Example ranking metric for a single random profile
    idx = np.random.randint(0, preds.shape[0])
    metrics = rank_metrics_for_profile(
        z_true=trues[idx],
        z_pred=preds[idx],
        up_thresh=cfg.data.up_threshold,
        down_thresh=cfg.data.down_threshold,
        k=50,
    )
    print("Ranking metrics for a random profile:", metrics)


if __name__ == "__main__":
    main()
