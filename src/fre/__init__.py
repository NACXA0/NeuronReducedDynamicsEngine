"""FRE: connectome-driven reduced neuron engine."""

from __future__ import annotations

from fre import models as _models  # noqa: F401
from fre.types import FittedActivation, GraphData, RateState, SpikeState

__version__ = "0.1.0"

__all__ = [
    "FittedActivation",
    "GraphData",
    "RateState",
    "SpikeState",
    "__version__",
]
