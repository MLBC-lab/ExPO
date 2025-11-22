from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, Sequence, Union, runtime_checkable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


PathLike = Union[str, Path]

if np is not None:
    ArrayLike = Union["np.ndarray", Sequence[float]]  # type: ignore[name-defined]
else:  # pragma: no cover
    ArrayLike = Sequence[float]


@runtime_checkable
class SupportsShape(Protocol):
    """Simple protocol for objects exposing a .shape tuple."""

    @property
    def shape(self) -> Sequence[int]:  # pragma: no cover - protocol
        ...


def ensure_path(path: PathLike) -> Path:
    """Return a Path instance for the given value.

    Accepts strings, Path objects, or any object implementing __fspath__.
    """
    if isinstance(path, Path):
        return path
    return Path(path)


def describe_array(x: ArrayLike | SupportsShape) -> str:
    """Return a compact string description of an array-like object."""
    if hasattr(x, "shape"):
        shape = getattr(x, "shape")
        return f"shape={tuple(shape)}"
    # Fall back to a simple length-based description
    try:
        n = len(x)  # type: ignore[arg-type]
    except Exception:
        return "scalar-like"
    return f"len={n}"
