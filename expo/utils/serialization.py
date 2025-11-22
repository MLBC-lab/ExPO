from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .typing_utils import PathLike


def save_model(
    path: PathLike,
    model: torch.nn.Module,
    extra: Dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_model(
    path: PathLike,
    model: torch.nn.Module,
    map_location: str | None = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    path = Path(path)
    state = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(state["model"])
    extra = {k: v for k, v in state.items() if k != "model"}
    return model, extra
