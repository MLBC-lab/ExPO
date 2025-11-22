from __future__ import annotations

from typing import Dict, Tuple


DEFAULT_NUM_GENES: int = 978
L1000_LANDMARK_COUNT: int = 978

# Reasonable bounds for exposure grids (hours, micromolar)
DEFAULT_TIME_MIN: float = 0.25
DEFAULT_TIME_MAX: float = 96.0
DEFAULT_DOSE_MIN: float = 0.01
DEFAULT_DOSE_MAX: float = 10.0

SMALL_EPS: float = 1e-8

# Thresholds for direction-of-change classification
DEFAULT_UP_THRESHOLD: float = 1.0
DEFAULT_DOWN_THRESHOLD: float = -1.0

# Recommended top-k values for ranking metrics used throughout the paper
DEFAULT_TOP_KS = (10, 25, 50, 100)

# Default set of ranking metrics reported in the manuscript
DEFAULT_RANKING_METRICS = ("ndcg", "jaccard", "rbo")

# Common subsets of landmark genes used in analysis
CANONICAL_P53_TARGETS = (
    "CDKN1A",
    "MDM2",
    "BAX",
    "BBC3",
    "GADD45A",
    "FAS",
)


def get_direction_thresholds() -> Tuple[float, float]:
    """Return the default (down, up) thresholds for regulation direction."""
    return DEFAULT_DOWN_THRESHOLD, DEFAULT_UP_THRESHOLD


def make_metric_config(
    up_threshold: float = DEFAULT_UP_THRESHOLD,
    down_threshold: float = DEFAULT_DOWN_THRESHOLD,
    top_ks: Tuple[int, ...] = DEFAULT_TOP_KS,
) -> Dict[str, object]:
    """Utility that centralizes choices for common metric settings.

    The returned dictionary is intentionally plain so that it can be serialized
    into experiment configs or JSON logs without additional processing.
    """
    return {
        "up_threshold": float(up_threshold),
        "down_threshold": float(down_threshold),
        "top_ks": tuple(int(k) for k in top_ks),
        "ranking_metrics": DEFAULT_RANKING_METRICS,
    }


def is_landmark_gene(symbol: str) -> bool:
    """Return True if the gene symbol is part of the L1000 landmark set.

    For now this simply checks that the symbol is uppercased; in a real system
    this would query the concrete landmark list. Keeping a dedicated helper
    function makes it easy to refine this semantics later.
    """
    if not symbol:
        return False
    return symbol.isupper()
