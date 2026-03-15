"""ExPO: Exposure–Profile Operator library.

This package implements the core model, training utilities, and analysis
tools used for dose–time–conditional transcriptomic prediction.
"""

__version__ = "0.1.0"
__author__ = "MLBC Lab"
__email__ = "info@mlbc-lab.org"

from .config import ExperimentConfig
from .models.expo_model import ExPOModel

__all__ = [
    "ExperimentConfig", 
    "ExPOModel",
    "__version__",
]
