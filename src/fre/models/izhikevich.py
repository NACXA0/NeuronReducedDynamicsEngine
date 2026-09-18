"""Izhikevich 2D reduction (R3 fallback when a 2D LUT is insufficient)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fre.models.registry import register


@dataclass
class IzhikevichParams:
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    Vth: float = 30.0
    t_ref: float = 0.0
    Vreset: float = -65.0


@register
class Izhikevich:
    name = "izhikevich"
    Params = IzhikevichParams
    reduction_benefit = "l2_fallback"

    @staticmethod
    def rhs(t: float, y: np.ndarray, I: float, p: IzhikevichParams) -> np.ndarray:
        v, u = y
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = p.a * (p.b * v - u)
        return np.array([dv, du], dtype=np.float64)

    @staticmethod
    def y0(p: IzhikevichParams) -> np.ndarray:
        return np.array([p.c, p.b * p.c], dtype=np.float64)

    @staticmethod
    def reset(y: np.ndarray, p: IzhikevichParams) -> np.ndarray:
        return np.array([p.c, y[1] + p.d], dtype=np.float64)

    @staticmethod
    def spike_threshold(p: IzhikevichParams) -> float:
        return p.Vth

    @staticmethod
    def hybrid_reset() -> bool:
        return True
