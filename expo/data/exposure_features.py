from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..constants import (
    DEFAULT_DOSE_MAX,
    DEFAULT_DOSE_MIN,
    DEFAULT_TIME_MAX,
    DEFAULT_TIME_MIN,
)


def log_transform_exposures(time: np.ndarray, dose: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    return np.log(time + eps), np.log(dose + eps)


def fourier_features(x: np.ndarray, num_frequencies: int) -> np.ndarray:
    x = x.reshape(-1, 1)
    freqs = np.arange(1, num_frequencies + 1, dtype=np.float32).reshape(1, -1)
    phases = x * freqs * math.pi
    return np.concatenate([np.sin(phases), np.cos(phases)], axis=1)


def build_exposure_features(
    times: np.ndarray,
    doses: np.ndarray,
    num_frequencies: int,
    log_eps: float,
    include_raw: bool = True,
) -> np.ndarray:
    t_log, d_log = log_transform_exposures(times, doses, log_eps)
    t_feat = fourier_features(t_log, num_frequencies)
    d_feat = fourier_features(d_log, num_frequencies)
    feats = np.concatenate([t_feat, d_feat], axis=1)
    if include_raw:
        feats = np.concatenate(
            [
                feats,
                times.reshape(-1, 1),
                doses.reshape(-1, 1),
                t_log.reshape(-1, 1),
                d_log.reshape(-1, 1),
            ],
            axis=1,
        )
    return feats.astype(np.float32)


@dataclass
class ExposureGrid:
    times: np.ndarray
    doses: np.ndarray

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.times, self.doses, indexing="ij")

    @classmethod
    def from_bounds(
        cls,
        t_min: float = DEFAULT_TIME_MIN,
        t_max: float = DEFAULT_TIME_MAX,
        d_min: float = DEFAULT_DOSE_MIN,
        d_max: float = DEFAULT_DOSE_MAX,
        t_steps: int = 16,
        d_steps: int = 16,
    ) -> "ExposureGrid":
        times = np.linspace(t_min, t_max, t_steps, dtype=np.float32)
        doses = np.linspace(d_min, d_max, d_steps, dtype=np.float32)
        return cls(times=times, doses=doses)

    def flatten(self) -> Tuple[np.ndarray, np.ndarray]:
        T, D = self.mesh()
        return T.ravel(), D.ravel()
