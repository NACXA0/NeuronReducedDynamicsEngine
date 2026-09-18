"""Shared hybrid Euler simulator for point-neuron models."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

import nrde.models as _models  # noqa: F401
from nrde.models.registry import NeuronModel, get_model
from nrde.sim_kernels import pack_model, simulate_spikes_packed, simulate_voltage_packed


def params_to_dict(params: Any) -> dict[str, float]:
    if hasattr(params, "__dataclass_fields__"):
        return {k: float(v) for k, v in asdict(params).items()}
    return {str(k): float(v) for k, v in dict(params).items()}


def resolve(model: str | NeuronModel, params: dict[str, float] | None = None):
    spec = get_model(model) if isinstance(model, str) else model
    p = spec.Params()
    if params:
        for k, v in params.items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
    return spec, p


def simulate_spikes(
    model: str | NeuronModel,
    params: dict[str, float] | None,
    I: float,
    t_total: float,
    dt: float = 0.05,
    t_discard: float = 0.0,
) -> np.ndarray:
    """Integrate one cell; return spike times in milliseconds."""
    spec, p = resolve(model, params)
    name = spec.name if not isinstance(model, str) else model
    packed = pack_model(name, p)
    if packed is not None:
        return simulate_spikes_packed(packed, float(I), t_total, dt, t_discard)
    return _simulate_spikes_generic(spec, p, float(I), t_total, dt, t_discard)


def _simulate_spikes_generic(
    spec: NeuronModel,
    p: Any,
    I: float,
    t_total: float,
    dt: float,
    t_discard: float,
) -> np.ndarray:
    n = int(np.round(t_total / dt))
    y = spec.y0(p).astype(np.float64)
    dy = np.empty_like(y)
    vth = spec.spike_threshold(p)
    t_ref = float(getattr(p, "t_ref", 0.0))
    ref_left = 0.0
    hybrid = spec.hybrid_reset()
    spikes: list[float] = []
    v_prev = float(y[0])
    rhs_into = getattr(spec, "rhs_into", None)
    for k in range(n):
        t = k * dt
        if rhs_into is not None:
            rhs_into(t, y, I, p, dy)
        else:
            np.copyto(dy, spec.rhs(t, y, I, p))
        if hybrid and ref_left > 0.0:
            ref_left = max(0.0, ref_left - dt)
            y += dt * dy
            y[0] = float(p.Vreset)
            v_prev = float(y[0])
            continue
        y += dt * dy
        v = float(y[0])
        spiked = v_prev < vth <= v
        if spiked:
            if t >= t_discard:
                spikes.append(t)
            if hybrid:
                y = spec.reset(y, p)
                ref_left = t_ref
                v = float(y[0])
        v_prev = v
    return np.asarray(spikes, dtype=np.float64)


def firing_rate(
    model: str | NeuronModel,
    params: dict[str, float] | None,
    I: float,
    t_total: float = 5000.0,
    dt: float = 0.05,
    window: float = 2000.0,
) -> float:
    t_discard = max(0.0, t_total - window)
    spikes = simulate_spikes(model, params, I, t_total, dt=dt, t_discard=t_discard)
    return float(spikes.size) / (window / 1000.0)


def euler_step(spec: NeuronModel, p: Any, y: np.ndarray, I: float, dt: float) -> np.ndarray:
    rhs_into = getattr(spec, "rhs_into", None)
    if rhs_into is not None:
        dy = np.empty_like(y)
        rhs_into(0.0, y, I, p, dy)
        return y + dt * dy
    dy = spec.rhs(0.0, y, I, p)
    return y + dt * dy


def simulate_voltage(
    model: str | NeuronModel,
    params: dict[str, float] | None,
    I: np.ndarray | float,
    dt: float = 0.05,
    clamp_spikes: bool = True,
) -> np.ndarray:
    """Euler integrate V(t). `I` is scalar or per-step current (nA)."""
    spec, p = resolve(model, params)
    I_arr = np.asarray(I, dtype=np.float64)
    if I_arr.ndim == 0:
        raise ValueError("Pass a current time series, or use simulate_spikes for constant I")
    name = spec.name if not isinstance(model, str) else model
    packed = pack_model(name, p)
    if packed is not None:
        return simulate_voltage_packed(packed, I_arr, dt, clamp_spikes)
    return _simulate_voltage_generic(spec, p, I_arr, dt, clamp_spikes)


def _simulate_voltage_generic(
    spec: NeuronModel,
    p: Any,
    I: np.ndarray,
    dt: float,
    clamp_spikes: bool,
) -> np.ndarray:
    y = spec.y0(p).astype(np.float64)
    dy = np.empty_like(y)
    vth = spec.spike_threshold(p)
    t_ref = float(getattr(p, "t_ref", 0.0))
    ref_left = 0.0
    hybrid = spec.hybrid_reset()
    V = np.empty(I.size, dtype=np.float64)
    v_prev = float(y[0])
    rhs_into = getattr(spec, "rhs_into", None)
    for k in range(I.size):
        Ik = float(I[k])
        t = k * dt
        if rhs_into is not None:
            rhs_into(t, y, Ik, p, dy)
        else:
            np.copyto(dy, spec.rhs(t, y, Ik, p))
        if hybrid and ref_left > 0.0:
            ref_left = max(0.0, ref_left - dt)
            y += dt * dy
            y[0] = float(p.Vreset)
            v_prev = float(y[0])
            V[k] = v_prev
            continue
        y += dt * dy
        v = float(y[0])
        if clamp_spikes and v_prev < vth <= v and hybrid:
            y = spec.reset(y, p)
            ref_left = t_ref
            v = float(y[0])
        v_prev = v
        V[k] = v
    return V
