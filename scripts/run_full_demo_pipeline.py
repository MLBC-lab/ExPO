from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def run(cmd, cwd=None) -> None:
    """Run a subprocess and stream its output, raising on failure."""
    print(f"[pipeline] Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def write_demo_config(config_path: Path, data_dir: Path, run_name: str = "demo_run") -> None:
    """Write a minimal ExperimentConfig-compatible JSON file."""
    cfg: Dict[str, Any] = {
        "data": {
            "expression_table": str(data_dir / "expression_table.pkl"),
            "metadata_table": str(data_dir / "metadata_table.pkl"),
            "compound_table": str(data_dir / "compound_table.pkl"),
            "split_scheme": "random",
            "n_folds": 5,
            "split_seed": 123,
            "up_threshold": 1.0,
            "down_threshold": -1.0,
        },
        "exposure": {
            "fourier_frequencies": 4,
            "log_eps": 1e-3,
            "include_raw_time_dose": True,
        },
        "chem": {
            "train_randomized_smiles": False,
            "randomized_smiles_prob": 0.0,
            "encoder_model_name": "chemberta-small",
            "freeze_encoder": True,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 32,
            "num_epochs": 3,
            "num_workers": 0,
            "seed": 42,
            "device": "cpu",
            "mixed_precision": False,
            "early_stopping_patience": 5,
            "warmup_steps": 0,
            "gradient_clip_norm": 1.0,
            "save_dir": "runs",
            "experiment_name": run_name,
        },
        "loss": {
            "huber_delta": 1.0,
            "listnet_temperature": 1.0,
            "sobolev_first": 0.0,
            "sobolev_second": 0.0,
            "lambda_regression": 1.0,
            "lambda_listnet": 0.0,
            "lambda_sobolev": 0.0,
            "lambda_monotonicity": 0.0,
            "lambda_quantile": 0.0,
        },
        "quantile": {
            "enabled": False,
            "quantiles": [0.1, 0.5, 0.9],
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def parse_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=str, default="demo_workspace")
    ap.add_argument("--keep-data", action="store_true", help="Do not delete data after run.")
    ap.add_argument("--n-profiles", type=int, default=512)
    ap.add_argument("--n-genes", type=int, default=200)
    return ap


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    cfg_dir = work_dir / "configs"
    run_name = "full_demo"

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate test data
    print("[pipeline] Step 1/4: generating test data")
    gen_cmd = [
        sys.executable,
        "scripts/generate_test_data.py",
        "--out-dir",
        str(data_dir),
        "--n-profiles",
        str(args.n_profiles),
        "--n-genes",
        str(args.n_genes),
    ]
    run(gen_cmd)

    # 2) Write demo config
    print("[pipeline] Step 2/4: writing configuration")
    cfg_path = cfg_dir / "demo_config.json"
    write_demo_config(cfg_path, data_dir, run_name=run_name)

    # 3) Train model
    print("[pipeline] Step 3/4: training model")
    train_cmd = [
        sys.executable,
        "scripts/train_expo.py",
        "--config",
        str(cfg_path),
    ]
    run(train_cmd)

    # 4) Generate training curves
    print("[pipeline] Step 4/4: generating plots")
    run_dir = Path("runs") / run_name
    plot_cmd = [
        sys.executable,
        "scripts/plot_training_curves.py",
        "--run-dir",
        str(run_dir),
    ]
    run(plot_cmd)

    print(f"[pipeline] Demo run completed. Results are in: {run_dir}")

    # Clean up data if requested
    if not args.keep_data and data_dir.exists():
        print(f"[pipeline] Removing data directory: {data_dir}")
        shutil.rmtree(data_dir, ignore_errors=True)
    else:
        print(f"[pipeline] Keeping data directory: {data_dir}")


if __name__ == "__main__":
    main()
