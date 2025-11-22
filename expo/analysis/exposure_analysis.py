from __future__ import annotations

from typing import Dict

import numpy as np

from ..data.exposure_features import ExposureGrid


def summarize_exposure_grid(grid: ExposureGrid) -> Dict[str, float]:
    T, D = grid.mesh()
    return {
        "time_min": float(T.min()),
        "time_max": float(T.max()),
        "dose_min": float(D.min()),
        "dose_max": float(D.max()),
        "n_points": float(T.size),
    }
