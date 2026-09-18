"""Neuron model registry (FR-2.3, DD-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class NeuronModel(Protocol):
    name: str
    Params: type

    @staticmethod
    def rhs(t: float, y: np.ndarray, I: float, p: object) -> np.ndarray: ...

    @staticmethod
    def y0(p: object) -> np.ndarray: ...

    @staticmethod
    def reset(y: np.ndarray, p: object) -> np.ndarray: ...

    @staticmethod
    def spike_threshold(p: object) -> float: ...

    @staticmethod
    def hybrid_reset() -> bool: ...

    reduction_benefit: str


_REGISTRY: dict[str, NeuronModel] = {}


def register(cls: NeuronModel) -> NeuronModel:
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str) -> NeuronModel:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(empty)"
        raise KeyError(f"Unknown neuron model {name!r}. Known: {known}") from exc


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def load_builtin_models() -> None:
    from nrde.models import adexp, explif, hh, izhikevich, lif  # noqa: F401


@dataclass
class _Sentinel:
    pass
