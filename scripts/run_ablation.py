import argparse
import copy
from pathlib import Path

from expo.config import ExperimentConfig
from expo.training import set_global_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    base_cfg = ExperimentConfig.load(args.config)

    ablations = {
        "no_basal": {"cell": {"use_basal_expression": False}},
        "no_listnet": {"loss": {"lambda_listnet": 0.0}},
        "no_sobolev": {"loss": {"lambda_sobolev": 0.0}},
    }

    for name, overrides in ablations.items():
        cfg_dict = base_cfg.to_dict()
        for section, changes in overrides.items():
            if section in cfg_dict and isinstance(cfg_dict[section], dict):
                cfg_dict[section].update(changes)
        cfg = ExperimentConfig.from_dict(cfg_dict)
        cfg.training.experiment_name = base_cfg.training.experiment_name + f"_{name}"
        set_global_seed(cfg.training.seed)
        out_path = Path(cfg.training.save_dir) / cfg.training.experiment_name / "config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.save(out_path)
        print(f"Wrote ablation config for {name} to {out_path}")


if __name__ == "__main__":
    main()
