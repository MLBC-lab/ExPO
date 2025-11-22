import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from expo.analysis import (
    plot_risk_coverage_curve,
    plot_calibration_curve,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="NPZ with fields: y_true, y_pred, errors, interval_widths")
    ap.add_argument("--output-dir", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    data = np.load(args.npz)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    errors = data["errors"]
    interval_widths = data["interval_widths"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Risk–coverage curve
    coverages = np.linspace(0.1, 1.0, 10)
    risks = []
    order = np.argsort(interval_widths)
    abs_err = np.abs(errors[order])
    n = len(abs_err)
    for c in coverages:
        k = max(1, int(round(c * n)))
        risks.append(float(abs_err[:k].mean()))

    fig, ax = plt.subplots()
    plot_risk_coverage_curve(
        coverages=np.asarray(coverages, dtype=np.float32),
        risks=np.asarray(risks, dtype=np.float32),
        title="Risk–coverage curve",
        ax=ax,
    )
    fig.savefig(out_dir / "risk_coverage.png", dpi=200)

    # Calibration curve
    fig2, ax2 = plt.subplots()
    plot_calibration_curve(
        y_true=y_true.flatten(),
        y_pred=y_pred.flatten(),
        n_bins=20,
        title="Calibration curve (flattened)",
        ax=ax2,
    )
    fig2.savefig(out_dir / "calibration_curve.png", dpi=200)

    print(f"Wrote uncertainty figures to {out_dir}")


if __name__ == "__main__":
    main()
