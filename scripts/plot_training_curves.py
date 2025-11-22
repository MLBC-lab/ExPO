import argparse
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt

from expo.analysis import plot_metric_curves, plot_error_histogram


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="Directory containing metrics.jsonl")
    ap.add_argument("--output-dir", type=str, required=False, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found")

    epochs = []
    train_mae = []
    val_mae = []

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            epochs.append(rec.get("epoch"))
            train_mae.append(rec.get("train_mae"))
            val_mae.append(rec.get("val_mae"))

    history = {
        "train_mae": train_mae,
        "val_mae": val_mae,
    }

    if args.output_dir is None:
        out_dir = run_dir / "figures"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot MAE curves
    fig, ax = plt.subplots()
    plot_metric_curves(history, x=epochs, title="MAE over epochs", ax=ax)
    fig.savefig(out_dir / "mae_curves.png", dpi=200)

    # Plot histogram of differences between train and val MAE
    diffs = np.asarray(train_mae) - np.asarray(val_mae)
    fig2, ax2 = plt.subplots()
    plot_error_histogram(np.abs(diffs), title="|train_mae - val_mae| histogram", ax=ax2)
    fig2.savefig(out_dir / "mae_gap_hist.png", dpi=200)

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
