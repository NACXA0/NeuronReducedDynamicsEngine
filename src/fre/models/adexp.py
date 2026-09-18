"""Adaptive exponential integrate-and-fire (AdExp / AdEx)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fre.models.registry import register


@dataclass
class AdExpParams:
    C: float = 0.2
    gL: float = 0.01
    EL: float = -70.0
    VT: float = -50.0
    DeltaT: float = 2.0
    a: float = 0.002
    tau_w: float = 100.0
    b: float = 0.06
    Vth: float = 0.0
    Vreset: float = -58.0
    t_ref: float = 2.0


@register
class AdExp:
    name = "adexp"
    Params = AdExpParams
    reduction_benefit = "v0.1_primary"

    @staticmethod
    def rhs(t: float, y: np.ndarray, I: float, p: AdExpParams) -> np.ndarray:
        v, w = y
        exp_term = p.gL * p.DeltaT * np.exp(np.clip((v - p.VT) / p.DeltaT, -20.0, 20.0))
        dv = (-p.gL * (v - p.EL) + exp_term - w + I) / p.C
        dw = (p.a * (v - p.EL) - w) / p.tau_w
        return np.array([dv, dw], dtype=np.float64)

    @staticmethod
    def y0(p: AdExpParams) -> np.ndarray:
        return np.array([p.EL, 0.0], dtype=np.float64)

    @staticmethod
    def reset(y: np.ndarray, p: AdExpParams) -> np.ndarray:
        return np.array([p.Vreset, y[1] + p.b], dtype=np.float64)

    @staticmethod
    def spike_threshold(p: AdExpParams) -> float:
        return p.Vth

    @staticmethod
    def hybrid_reset() -> bool:
        return True
