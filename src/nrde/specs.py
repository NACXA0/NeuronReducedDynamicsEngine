"""Public ModelSpec protocol (RFC-002 §3.3 protocol 1).

Built-in LIF / ExpLIF / AdExp / HH / Izhikevich are reference implementations.
Breaking changes require an RFC and keep one version of backward compatibility (R11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from nrde.models.registry import NeuronModel, get_model, list_models, load_builtin_models, register

PROTOCOL_VERSION = 1


@dataclass
class ModelSpec:
    """Upload / registration interface for third-party ODEs (C-layer)."""

    name: str
    state_dims: int
    rhs: Callable[[float, np.ndarray, float, Any], np.ndarray]
    y0: Callable[[Any], np.ndarray]
    Params: type = dict  # type: ignore[assignment]
    fitted_targets: list[str] = field(default_factory=lambda: ["f-i"])
    reset: Callable[[np.ndarray, Any], np.ndarray] | None = None
    spike_threshold: Callable[[Any], float] | None = None
    hybrid_reset: bool = True
    reduction_benefit: str = "external"
    protocol_version: int = PROTOCOL_VERSION

    def register(self) -> NeuronModel:
        """Install this spec into the global model registry."""
        return register(_spec_as_model(self))


def _spec_as_model(spec: ModelSpec) -> NeuronModel:
    class _Dynamic:
        name = spec.name
        Params = spec.Params
        reduction_benefit = spec.reduction_benefit

        @staticmethod
        def rhs(t: float, y: np.ndarray, I: float, p: object) -> np.ndarray:
            return np.asarray(spec.rhs(t, y, I, p), dtype=np.float64)

        @staticmethod
        def y0(p: object) -> np.ndarray:
            return np.asarray(spec.y0(p), dtype=np.float64)

        @staticmethod
        def reset(y: np.ndarray, p: object) -> np.ndarray:
            if spec.reset is not None:
                return np.asarray(spec.reset(y, p), dtype=np.float64)
            out = np.asarray(y, dtype=np.float64).copy()
            out[0] = float(getattr(p, "Vreset", getattr(p, "V_reset", getattr(p, "Vr", -70.0))))
            return out

        @staticmethod
        def spike_threshold(p: object) -> float:
            if spec.spike_threshold is not None:
                return float(spec.spike_threshold(p))
            return float(getattr(p, "Vth", getattr(p, "V_th", -50.0)))

        @staticmethod
        def hybrid_reset() -> bool:
            return bool(spec.hybrid_reset)

    return _Dynamic  # type: ignore[return-value]


def model_spec_from_registry(name: str) -> ModelSpec:
    """Wrap a registered NeuronModel as a ModelSpec (documentation / introspection)."""
    load_builtin_models()
    model = get_model(name)
    y0_probe = np.asarray(model.y0(model.Params()), dtype=np.float64)
    return ModelSpec(
        name=model.name,
        state_dims=int(y0_probe.size),
        rhs=model.rhs,
        y0=model.y0,
        Params=model.Params,
        fitted_targets=["f-i", "kernels", "H_omega"],
        reset=model.reset,
        spike_threshold=model.spike_threshold,
        hybrid_reset=bool(model.hybrid_reset()),
        reduction_benefit=str(getattr(model, "reduction_benefit", "control")),
    )


def list_specs() -> Sequence[str]:
    load_builtin_models()
    return list_models()
