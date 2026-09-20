"""NRDE: type-bound reduced-dynamics engine (RFC-002 surface; RFC-003 intent)."""

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
