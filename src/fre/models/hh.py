"""Point-neuron Hodgkin-Huxley (1e-4 cm^2 membrane so I is in nA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fre.models.registry import register


def _alpha_m(v: float) -> float:
    d = v + 40.0
    if abs(d) < 1e-6:
        return 1.0
    return 0.1 * d / (1.0 - np.exp(-d / 10.0))


def _beta_m(v: float) -> float:
    return 4.0 * np.exp(-(v + 65.0) / 18.0)


def _alpha_h(v: float) -> float:
    return 0.07 * np.exp(-(v + 65.0) / 20.0)


def _beta_h(v: float) -> float:
    return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))


def _alpha_n(v: float) -> float:
    d = v + 55.0
    if abs(d) < 1e-6:
        return 0.1
    return 0.01 * d / (1.0 - np.exp(-d / 10.0))


def _beta_n(v: float) -> float:
    return 0.125 * np.exp(-(v + 65.0) / 80.0)


@dataclass
class HHParams:
    C: float = 0.1
    gNa: float = 12.0
    gK: float = 3.6
    gL: float = 0.03
    ENa: float = 50.0
    EK: float = -77.0
    EL: float = -54.387
    Vth: float = 0.0
    t_ref: float = 0.0


@register
class HH:
    name = "hh"
    Params = HHParams
    reduction_benefit = "v0.2"

    @staticmethod
    def rhs(t: float, y: np.ndarray, I: float, p: HHParams) -> np.ndarray:
        v, m, h, n = y
        i_na = p.gNa * (m**3) * h * (v - p.ENa)
        i_k = p.gK * (n**4) * (v - p.EK)
        i_l = p.gL * (v - p.EL)
        dv = (I - i_na - i_k - i_l) / p.C
        am, bm = _alpha_m(v), _beta_m(v)
        ah, bh = _alpha_h(v), _beta_h(v)
        an, bn = _alpha_n(v), _beta_n(v)
        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n
        return np.array([dv, dm, dh, dn], dtype=np.float64)

    @staticmethod
    def y0(p: HHParams) -> np.ndarray:
        v = -65.0
        am, bm = _alpha_m(v), _beta_m(v)
        ah, bh = _alpha_h(v), _beta_h(v)
        an, bn = _alpha_n(v), _beta_n(v)
        return np.array(
            [v, am / (am + bm), ah / (ah + bh), an / (an + bn)],
            dtype=np.float64,
        )

    @staticmethod
    def reset(y: np.ndarray, p: HHParams) -> np.ndarray:
        return np.array(y, dtype=np.float64, copy=True)

    @staticmethod
    def spike_threshold(p: HHParams) -> float:
        return p.Vth

    @staticmethod
    def hybrid_reset() -> bool:
        return False
