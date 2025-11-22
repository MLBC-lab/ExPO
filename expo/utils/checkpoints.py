from __future__ import annotations

from pathlib import Path
from typing import List

from .typing_utils import PathLike


def list_checkpoints(pattern: PathLike) -> List[Path]:
    path = Path(pattern)
    if path.is_file():
        return [path]
    # Interpret pattern as directory
    if path.is_dir():
        return sorted(path.glob("*.pt"))
    # Treat as glob pattern
    return sorted(Path().glob(str(path)))
