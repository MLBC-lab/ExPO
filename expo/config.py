from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .constants import (
    DEFAULT_UP_THRESHOLD,
    DEFAULT_DOWN_THRESHOLD,
)


@dataclass
class DataConfig:
    expression_table: str
    metadata_table: str
    compound_table: str
    basal_expression_table: Optional[str] = None

    up_threshold: float = DEFAULT_UP_THRESHOLD
    down_threshold: float = DEFAULT_DOWN_THRESHOLD

    log_eps: float = 1e-3
    fourier_frequencies: int = 16

    n_folds: int = 5
    split_seed: int = 42
    split_scheme: str = "scaffold"  # scaffold|random|leave_cell

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChemConfig:
    pretrained_model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    max_smiles_length: int = 256
    freeze_lower_layers: int = 6
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    train_randomized_smiles: bool = True
    randomized_smiles_prob: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CellConfig:
    num_cells: int = 0
    embedding_dim: int = 512
    use_basal_expression: bool = True
    basal_projection_dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExposureConfig:
    log_eps: float = 1e-3
    fourier_frequencies: int = 16
    include_raw_time_dose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OperatorConfig:
    r: int = 256
    branch_hidden: Tuple[int, int] = (512, 256)
    trunk_hidden: Tuple[int, int] = (256, 256)
    context_dim: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantileConfig:
    enabled: bool = True
    quantiles: Tuple[float, float] = (0.1, 0.9)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LossConfig:
    huber_delta: float = 1.0
    listnet_temperature: float = 0.5
    lambda_regression: float = 1.0
    lambda_listnet: float = 1.0
    lambda_sobolev: float = 1e-2
    lambda_monotonicity: float = 1e-2
    lambda_quantile: float = 1.0
    sobolev_first: float = 1.0
    sobolev_second: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 17
    save_dir: str = "runs"
    experiment_name: str = "expo_default"
    mixed_precision: bool = True
    early_stopping_patience: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    data: DataConfig
    chem: ChemConfig = field(default_factory=ChemConfig)
    cell: CellConfig = field(default_factory=CellConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    quantile: QuantileConfig = field(default_factory=QuantileConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ExperimentConfig":
        data = DataConfig(**cfg["data"])
        chem = ChemConfig(**cfg.get("chem", {}))
        cell = CellConfig(**cfg.get("cell", {}))
        exposure = ExposureConfig(**cfg.get("exposure", {}))
        operator = OperatorConfig(**cfg.get("operator", {}))
        quantile = QuantileConfig(**cfg.get("quantile", {}))
        loss = LossConfig(**cfg.get("loss", {}))
        training = TrainingConfig(**cfg.get("training", {}))
        extra = {
            k: v
            for k, v in cfg.items()
            if k
            not in {
                "data",
                "chem",
                "cell",
                "exposure",
                "operator",
                "quantile",
                "loss",
                "training",
            }
        }
        return cls(
            data=data,
            chem=chem,
            cell=cell,
            exposure=exposure,
            operator=operator,
            quantile=quantile,
            loss=loss,
            training=training,
            extra=extra,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        return cls.load(path)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if path.suffix.lower() in {".yml", ".yaml"}:
            import yaml  # type: ignore[import]

            cfg_dict = yaml.safe_load(path.read_text())
        else:
            cfg_dict = json.loads(path.read_text())
        return cls.from_dict(cfg_dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".yml", ".yaml"}:
            import yaml  # type: ignore[import]

            yaml.safe_dump(self.to_dict(), path.open("w"), sort_keys=False)
        else:
            json.dump(self.to_dict(), path.open("w"), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data.to_dict(),
            "chem": self.chem.to_dict(),
            "cell": self.cell.to_dict(),
            "exposure": self.exposure.to_dict(),
            "operator": self.operator.to_dict(),
            "quantile": self.quantile.to_dict(),
            "loss": self.loss.to_dict(),
            "training": self.training.to_dict(),
            **self.extra,
        }

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        """Create a modified copy with a shallow override of top-level sections."""
        cfg = self.to_dict()
        cfg.update(kwargs)
        return ExperimentConfig.from_dict(cfg)
