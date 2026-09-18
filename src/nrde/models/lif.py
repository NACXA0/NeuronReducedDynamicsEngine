"""Leaky integrate-and-fire (current-based). Units: mV, ms, nA, nF, µS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nrde.models.registry import register


@dataclass
class LIFParams:
    C: float = 0.1
    gL: float = 0.01
    EL: float = -70.0
    Vth: float = -50.0
    Vreset: float = -65.0
    t_ref: float = 2.0


@register
class LIF:
    name = "lif"
    Params = LIFParams
    reduction_benefit = "control"

    @staticmethod
    def rhs(t: float, y: np.ndarray, I: float, p: LIFParams) -> np.ndarray:
        v = y[0]
        dv = (-p.gL * (v - p.EL) + I) / p.C
        return np.array([dv], dtype=np.float64)

    @staticmethod
    def y0(p: LIFParams) -> np.ndarray:
        return np.array([p.EL], dtype=np.float64)

    @staticmethod
    def reset(y: np.ndarray, p: LIFParams) -> np.ndarray:
        out = np.array(y, dtype=np.float64, copy=True)
        out[0] = p.Vreset
        return out

    @staticmethod
    def spike_threshold(p: LIFParams) -> float:
        return p.Vth

    @staticmethod
    def hybrid_reset() -> bool:
        return True


def analytic_rate(I: float, p: LIFParams | None = None) -> float:
    """Exact LIF f-I for constant current (Hz)."""
    p = p or LIFParams()
    v_inf = p.EL + I / p.gL
    if v_inf <= p.Vth:
        return 0.0
    tau = p.C / p.gL
    num = v_inf - p.Vreset
    den = v_inf - p.Vth
    if den <= 0.0 or num <= 0.0:
        return 0.0
    isi = p.t_ref + tau * np.log(num / den)
    if isi <= 0.0:
        return 0.0
    return 1000.0 / isi
