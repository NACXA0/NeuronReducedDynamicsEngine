"""Scalar Euler kernels for built-in models.

Python fallback is allocation-free; Numba is used when installed (optional extra).
Custom ModelSpec types keep the generic `spec.rhs` path in `nrde.sim`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# pvec layouts (unused slots stay 0):
# LIF        0  C, gL, EL, Vth, Vreset, t_ref
# ExpLIF     1  C, gL, EL, VT, DeltaT, Vth, Vreset, t_ref
# AdExp      2  C, gL, EL, VT, DeltaT, a, tau_w, b, Vth, Vreset, t_ref
# Izhikevich 3  a, b, c, d, Vth, t_ref
# HH         4  C, gNa, gK, gL, ENa, EK, EL, Vth
MODEL_LIF = 0
MODEL_EXPLIF = 1
MODEL_ADEXP = 2
MODEL_IZH = 3
MODEL_HH = 4
_PVEC = 12


try:
    from numba import prange as _prange
except Exception:  # pragma: no cover - Numba optional
    _prange = range


def _maybe_njit(fn):
    try:
        from numba import njit

        return njit(cache=True, nogil=True)(fn)
    except Exception:
        return fn


def _maybe_njit_parallel(fn):
    try:
        from numba import njit

        return njit(cache=True, nogil=True, parallel=True)(fn)
    except Exception:
        return fn


@dataclass(frozen=True)
class PackedModel:
    model_id: int
    p: np.ndarray
    y0: np.ndarray
    n_state: int
    vth: float
    t_ref: float
    hybrid: bool
    vreset: float


def pack_model(name: str, params: Any) -> PackedModel | None:
    p = np.zeros(_PVEC, dtype=np.float64)
    if name == "lif":
        p[0:6] = (params.C, params.gL, params.EL, params.Vth, params.Vreset, params.t_ref)
        y0 = np.array([params.EL], dtype=np.float64)
        return PackedModel(MODEL_LIF, p, y0, 1, params.Vth, params.t_ref, True, params.Vreset)
    if name == "explif":
        p[0:8] = (
            params.C,
            params.gL,
            params.EL,
            params.VT,
            params.DeltaT,
            params.Vth,
            params.Vreset,
            params.t_ref,
        )
        y0 = np.array([params.EL], dtype=np.float64)
        return PackedModel(MODEL_EXPLIF, p, y0, 1, params.Vth, params.t_ref, True, params.Vreset)
    if name == "adexp":
        p[0:11] = (
            params.C,
            params.gL,
            params.EL,
            params.VT,
            params.DeltaT,
            params.a,
            params.tau_w,
            params.b,
            params.Vth,
            params.Vreset,
            params.t_ref,
        )
        y0 = np.array([params.EL, 0.0], dtype=np.float64)
        return PackedModel(MODEL_ADEXP, p, y0, 2, params.Vth, params.t_ref, True, params.Vreset)
    if name == "izhikevich":
        p[0:6] = (params.a, params.b, params.c, params.d, params.Vth, params.t_ref)
        y0 = np.array([params.c, params.b * params.c], dtype=np.float64)
        return PackedModel(MODEL_IZH, p, y0, 2, params.Vth, params.t_ref, True, params.c)
    if name == "hh":
        p[0:8] = (params.C, params.gNa, params.gK, params.gL, params.ENa, params.EK, params.EL, params.Vth)
        v = -65.0
        am, bm = _alpha_m(v), _beta_m(v)
        ah, bh = _alpha_h(v), _beta_h(v)
        an, bn = _alpha_n(v), _beta_n(v)
        y0 = np.array([v, am / (am + bm), ah / (ah + bh), an / (an + bn)], dtype=np.float64)
        return PackedModel(MODEL_HH, p, y0, 4, params.Vth, float(getattr(params, "t_ref", 0.0)), False, 0.0)
    return None


def _alpha_m(v: float) -> float:
    d = v + 40.0
    if abs(d) < 1e-6:
        return 1.0
    return 0.1 * d / (1.0 - math.exp(-d / 10.0))


def _beta_m(v: float) -> float:
    return 4.0 * math.exp(-(v + 65.0) / 18.0)


def _alpha_h(v: float) -> float:
    return 0.07 * math.exp(-(v + 65.0) / 20.0)


def _beta_h(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))


def _alpha_n(v: float) -> float:
    d = v + 55.0
    if abs(d) < 1e-6:
        return 0.1
    return 0.01 * d / (1.0 - math.exp(-d / 10.0))


def _beta_n(v: float) -> float:
    return 0.125 * math.exp(-(v + 65.0) / 80.0)


@_maybe_njit
def _rhs_core(model_id: int, y: np.ndarray, I: float, p: np.ndarray, dy: np.ndarray) -> None:
    if model_id == 0:
        dy[0] = (-p[1] * (y[0] - p[2]) + I) / p[0]
    elif model_id == 1:
        x = (y[0] - p[3]) / p[4]
        if x > 20.0:
            x = 20.0
        elif x < -20.0:
            x = -20.0
        exp_term = p[1] * p[4] * math.exp(x)
        dy[0] = (-p[1] * (y[0] - p[2]) + exp_term + I) / p[0]
    elif model_id == 2:
        x = (y[0] - p[3]) / p[4]
        if x > 20.0:
            x = 20.0
        elif x < -20.0:
            x = -20.0
        exp_term = p[1] * p[4] * math.exp(x)
        dy[0] = (-p[1] * (y[0] - p[2]) + exp_term - y[1] + I) / p[0]
        dy[1] = (p[5] * (y[0] - p[2]) - y[1]) / p[6]
    elif model_id == 3:
        dy[0] = 0.04 * y[0] * y[0] + 5.0 * y[0] + 140.0 - y[1] + I
        dy[1] = p[0] * (p[1] * y[0] - y[1])
    else:
        v = y[0]
        m = y[1]
        h = y[2]
        n = y[3]
        i_na = p[1] * (m * m * m) * h * (v - p[4])
        i_k = p[2] * (n * n * n * n) * (v - p[5])
        i_l = p[3] * (v - p[6])
        dy[0] = (I - i_na - i_k - i_l) / p[0]
        d = v + 40.0
        am = 1.0 if abs(d) < 1e-6 else 0.1 * d / (1.0 - math.exp(-d / 10.0))
        bm = 4.0 * math.exp(-(v + 65.0) / 18.0)
        ah = 0.07 * math.exp(-(v + 65.0) / 20.0)
        bh = 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))
        d2 = v + 55.0
        an = 0.1 if abs(d2) < 1e-6 else 0.01 * d2 / (1.0 - math.exp(-d2 / 10.0))
        bn = 0.125 * math.exp(-(v + 65.0) / 80.0)
        dy[1] = am * (1.0 - m) - bm * m
        dy[2] = ah * (1.0 - h) - bh * h
        dy[3] = an * (1.0 - n) - bn * n


@_maybe_njit
def _reset_core(model_id: int, y: np.ndarray, p: np.ndarray, vreset: float) -> None:
    if model_id == 2:
        y[0] = p[9]
        y[1] = y[1] + p[7]
    elif model_id == 3:
        y[0] = p[2]
        y[1] = y[1] + p[3]
    elif model_id != 4:
        y[0] = vreset


@_maybe_njit
def _simulate_spikes_core(
    model_id: int,
    p: np.ndarray,
    y0: np.ndarray,
    n_state: int,
    I: float,
    dt: float,
    n: int,
    t_discard: float,
    vth: float,
    t_ref: float,
    hybrid: bool,
    vreset: float,
) -> np.ndarray:
    y = y0.copy()
    dy = np.zeros(n_state, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)
    n_spk = 0
    v_prev = y[0]
    ref_left = 0.0
    for k in range(n):
        t = k * dt
        if hybrid and ref_left > 0.0:
            ref_left = ref_left - dt
            if ref_left < 0.0:
                ref_left = 0.0
            _rhs_core(model_id, y, I, p, dy)
            for s in range(n_state):
                y[s] = y[s] + dt * dy[s]
            y[0] = vreset
            v_prev = y[0]
            continue
        _rhs_core(model_id, y, I, p, dy)
        for s in range(n_state):
            y[s] = y[s] + dt * dy[s]
        v = y[0]
        spiked = v_prev < vth <= v
        if spiked:
            if t >= t_discard:
                times[n_spk] = t
                n_spk += 1
            if hybrid:
                _reset_core(model_id, y, p, vreset)
                ref_left = t_ref
                v = y[0]
        v_prev = v
    return times[:n_spk].copy()


@_maybe_njit
def _spike_count_core(
    model_id: int,
    p: np.ndarray,
    y0: np.ndarray,
    n_state: int,
    I: float,
    dt: float,
    n: int,
    t_discard: float,
    vth: float,
    t_ref: float,
    hybrid: bool,
    vreset: float,
) -> int:
    y = y0.copy()
    dy = np.zeros(n_state, dtype=np.float64)
    n_spk = 0
    v_prev = y[0]
    ref_left = 0.0
    for k in range(n):
        t = k * dt
        if hybrid and ref_left > 0.0:
            ref_left = ref_left - dt
            if ref_left < 0.0:
                ref_left = 0.0
            _rhs_core(model_id, y, I, p, dy)
            for s in range(n_state):
                y[s] = y[s] + dt * dy[s]
            y[0] = vreset
            v_prev = y[0]
            continue
        _rhs_core(model_id, y, I, p, dy)
        for s in range(n_state):
            y[s] = y[s] + dt * dy[s]
        v = y[0]
        spiked = v_prev < vth <= v
        if spiked:
            if t >= t_discard:
                n_spk += 1
            if hybrid:
                _reset_core(model_id, y, p, vreset)
                ref_left = t_ref
                v = y[0]
        v_prev = v
    return n_spk


@_maybe_njit_parallel
def _firing_rates_batch_core(
    model_id: int,
    p: np.ndarray,
    y0: np.ndarray,
    n_state: int,
    I_grid: np.ndarray,
    dt: float,
    n: int,
    t_discard: float,
    vth: float,
    t_ref: float,
    hybrid: bool,
    vreset: float,
    window_s: float,
) -> np.ndarray:
    n_I = I_grid.size
    rates = np.empty(n_I, dtype=np.float64)
    for i in _prange(n_I):
        n_spk = _spike_count_core(
            model_id,
            p,
            y0,
            n_state,
            I_grid[i],
            dt,
            n,
            t_discard,
            vth,
            t_ref,
            hybrid,
            vreset,
        )
        rates[i] = n_spk / window_s
    return rates


@_maybe_njit
def _simulate_voltage_core(
    model_id: int,
    p: np.ndarray,
    y0: np.ndarray,
    n_state: int,
    I: np.ndarray,
    dt: float,
    vth: float,
    t_ref: float,
    hybrid: bool,
    vreset: float,
    clamp_spikes: bool,
) -> np.ndarray:
    n = I.size
    y = y0.copy()
    dy = np.zeros(n_state, dtype=np.float64)
    V = np.empty(n, dtype=np.float64)
    v_prev = y[0]
    ref_left = 0.0
    for k in range(n):
        Ik = I[k]
        if hybrid and ref_left > 0.0:
            ref_left = ref_left - dt
            if ref_left < 0.0:
                ref_left = 0.0
            _rhs_core(model_id, y, Ik, p, dy)
            for s in range(n_state):
                y[s] = y[s] + dt * dy[s]
            y[0] = vreset
            v_prev = y[0]
            V[k] = v_prev
            continue
        _rhs_core(model_id, y, Ik, p, dy)
        for s in range(n_state):
            y[s] = y[s] + dt * dy[s]
        v = y[0]
        if clamp_spikes and v_prev < vth <= v and hybrid:
            _reset_core(model_id, y, p, vreset)
            ref_left = t_ref
            v = y[0]
        v_prev = v
        V[k] = v
    return V


@_maybe_njit_parallel
def _fill_spike_lut(
    model_id: int,
    p: np.ndarray,
    y0: np.ndarray,
    n_state: int,
    I_axis: np.ndarray,
    dt_axis: np.ndarray,
    dt: float,
    sub: float,
    vth: float,
    t_ref: float,
    hybrid: bool,
    vreset: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_I = I_axis.size
    n_dt = dt_axis.size
    spike_mask = np.zeros((n_I, n_dt), dtype=np.float32)
    V_next = np.zeros((n_I, n_dt), dtype=np.float32)
    hits = 0
    for i in _prange(n_I):
        dy = np.zeros(n_state, dtype=np.float64)
        I_val = I_axis[i]
        for j in range(n_dt):
            t_since = dt_axis[j]
            y = y0.copy()
            if hybrid:
                _reset_core(model_id, y, p, vreset)
            elapsed = 0.0
            while elapsed + 1e-12 < t_since:
                in_ref = hybrid and elapsed < t_ref
                I_now = 0.0 if in_ref else I_val
                _rhs_core(model_id, y, I_now, p, dy)
                for s in range(n_state):
                    y[s] = y[s] + sub * dy[s]
                if in_ref:
                    y[0] = vreset
                elapsed += sub
            v_prev = y[0]
            in_ref_now = hybrid and t_since < t_ref
            I_now = 0.0 if in_ref_now else I_val
            _rhs_core(model_id, y, I_now, p, dy)
            for s in range(n_state):
                y[s] = y[s] + dt * dy[s]
            crossed = v_prev < vth <= y[0]
            already_high = (not in_ref_now) and v_prev >= vth
            spiked = (not in_ref_now) and (crossed or already_high)
            if spiked and hybrid:
                _reset_core(model_id, y, p, vreset)
            elif in_ref_now and hybrid:
                y[0] = vreset
            spike_mask[i, j] = 1.0 if spiked else 0.0
            V_next[i, j] = y[0]
            if spiked:
                hits += 1
    return spike_mask, V_next, hits


def simulate_spikes_packed(
    packed: PackedModel,
    I: float,
    t_total: float,
    dt: float,
    t_discard: float,
) -> np.ndarray:
    n = int(np.round(t_total / dt))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return _simulate_spikes_core(
        packed.model_id,
        packed.p,
        packed.y0,
        packed.n_state,
        float(I),
        float(dt),
        n,
        float(t_discard),
        packed.vth,
        packed.t_ref,
        packed.hybrid,
        packed.vreset,
    )


def simulate_voltage_packed(
    packed: PackedModel,
    I: np.ndarray,
    dt: float,
    clamp_spikes: bool,
) -> np.ndarray:
    return _simulate_voltage_core(
        packed.model_id,
        packed.p,
        packed.y0,
        packed.n_state,
        np.asarray(I, dtype=np.float64),
        float(dt),
        packed.vth,
        packed.t_ref,
        packed.hybrid,
        packed.vreset,
        bool(clamp_spikes),
    )


def firing_rates_packed(
    packed: PackedModel,
    I_grid: np.ndarray,
    t_total: float,
    dt: float,
    window: float,
) -> np.ndarray:
    """Vector of firing rates (Hz) for each current in `I_grid`."""
    n = int(np.round(t_total / dt))
    I_grid = np.ascontiguousarray(I_grid, dtype=np.float64)
    if n <= 0 or I_grid.size == 0:
        return np.zeros(I_grid.size, dtype=np.float64)
    t_discard = max(0.0, float(t_total) - float(window))
    window_s = max(float(window) / 1000.0, 1e-12)
    return _firing_rates_batch_core(
        packed.model_id,
        packed.p,
        packed.y0,
        packed.n_state,
        I_grid,
        float(dt),
        n,
        t_discard,
        packed.vth,
        packed.t_ref,
        packed.hybrid,
        packed.vreset,
        window_s,
    )


def fill_spike_lut_packed(
    packed: PackedModel,
    I_axis: np.ndarray,
    dt_axis: np.ndarray,
    dt: float,
    sub: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    return _fill_spike_lut(
        packed.model_id,
        packed.p,
        packed.y0,
        packed.n_state,
        np.asarray(I_axis, dtype=np.float64),
        np.asarray(dt_axis, dtype=np.float64),
        float(dt),
        float(sub),
        packed.vth,
        packed.t_ref,
        packed.hybrid,
        packed.vreset,
    )
