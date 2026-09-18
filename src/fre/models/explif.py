"""Exponential LIF (Brette-Gerstner without adaptation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fre.models.registry import register


@dataclass
class ExpLIFParams:
    C: float = 0.1
    gL: float = 0.01
    EL: float = -70.0
    VT: float = -50.0
    DeltaT: float = 2.0
    Vth: float = 0.0
    Vreset: float = -65.0
    t_ref: float = 2.0


@register
class ExpLIF:
    name = "explif"
    Params = ExpLIFParams
    reduction_benefit = "control"

    @staticmethod
    def rhs(t: float, y: np.ndarray, I: float, p: ExpLIFParams) -> np.ndarray:
        v = y[0]
        exp_term = p.gL * p.DeltaT * np.exp(np.clip((v - p.VT) / p.DeltaT, -20.0, 20.0))
        dv = (-p.gL * (v - p.EL) + exp_term + I) / p.C
        return np.array([dv], dtype=np.float64)

    @staticmethod
    def y0(p: ExpLIFParams) -> np.ndarray:
        return np.array([p.EL], dtype=np.float64)

    @staticmethod
    def reset(y: np.ndarray, p: ExpLIFParams) -> np.ndarray:
        out = np.array(y, dtype=np.float64, copy=True)
        out[0] = p.Vreset
        return out

    @staticmethod
    def spike_threshold(p: ExpLIFParams) -> float:
        return p.Vth

    @staticmethod
    def hybrid_reset() -> bool:
        return True
