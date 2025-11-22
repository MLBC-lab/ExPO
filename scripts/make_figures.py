import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from expo.analysis import plot_dose_time_map
from expo.data.exposure_features import ExposureGrid
from expo.utils.io import load_table


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expression-table", type=str, required=True)
    ap.add_argument("--metadata-table", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    expr_df = load_table(args.expression_table)
    meta_df = load_table(args.metadata_table)

    # Example: visualize mean absolute expression change for a single gene
    gene = expr_df.columns[0]
    merged = meta_df.join(expr_df[[gene]])

    grid = ExposureGrid.from_bounds(
        t_min=float(merged["time"].min()),
        t_max=float(merged["time"].max()),
        d_min=float(merged["dose"].min()),
        d_max=float(merged["dose"].max()),
        t_steps=24,
        d_steps=24,
    )
    T, D = grid.mesh()
    values = np.zeros_like(T, dtype=np.float32)
    # Very simple aggregation just for visualization
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            mask = (merged["time"] == T[i, j]) & (merged["dose"] == D[i, j])
            if mask.any():
                values[i, j] = merged.loc[mask, gene].abs().mean()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"{gene}_dose_time_map.png"
    plot_dose_time_map(
        values=values,
        times=grid.times,
        doses=grid.doses,
        title=f"Dose–time map for {gene}",
    )
    import matplotlib.pyplot as plt

    plt.savefig(fig_path, dpi=200)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
