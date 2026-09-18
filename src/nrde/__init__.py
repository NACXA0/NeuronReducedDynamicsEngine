"""NRDE: connectome-driven reduced neuron engine (RFC-002 public surface)."""

from __future__ import annotations

from nrde.cli import offline, run
from nrde.presets import demo, make
from nrde.specs import ModelSpec

__version__ = "0.1.0"

__all__ = [
    "ModelSpec",
    "demo",
    "make",
    "offline",
    "run",
    "__version__",
]
