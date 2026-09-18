"""Offline f-I and 2D LUT fitting (FR-3, DD-3)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from fre import models as _models  # noqa: F401
from fre.offline.chirp import chirp_impedance, decide_layer
from fre.sim import firing_rate, resolve
from fre.types import NPZ_SCHEMA_VERSION, ExpKernel, FittedActivation, SRMKernels

R2_GATE = 0.98


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-18:
        return 1.0 if ss_res <= 1e-18 else 0.0
    return 1.0 - ss_res / ss_tot


def _onset_current(I_grid: np.ndarray, r_grid: np.ndarray, min_hz: float = 1.0) -> float:
    firing = np.where(r_grid >= min_hz)[0]
    if firing.size == 0:
        return float(I_grid[-1] + 1.0)
    return float(I_grid[firing[0]])


def _spline_overshoots(I_grid: np.ndarray, r_grid: np.ndarray) -> bool:
    if I_grid.size < 4:
        return False
    r_max = float(np.max(r_grid) + 1e-9)
    try:
        cs = CubicSpline(I_grid, r_grid)
    except ValueError:
        return True
    dense = np.linspace(I_grid[0], I_grid[-1], 256)
    r = cs(dense)
    if np.any(r < -0.5):
        return True
    if np.any(r > 1.05 * r_max):
        return True
    onset = _onset_current(I_grid, r_grid)
    pre = dense < onset
    if np.any(pre) and np.any(r[pre] > 5.0) and float(r_grid[0]) < 1.0:
        return True
    return False


def scan_fi_curve(
    model: str,
    params: dict[str, float] | None = None,
    I_min: float = 0.0,
    I_max: float = 1.0,
    n_I: int = 64,
    t_total: float = 5000.0,
    window: float = 2000.0,
    dt: float = 0.05,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    I_grid = np.linspace(I_min, I_max, n_I)

    def _one(I: float) -> float:
        return firing_rate(model, params, float(I), t_total=t_total, dt=dt, window=window)

    if n_jobs and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            rates = list(pool.map(_one, I_grid.tolist()))
        r_grid = np.asarray(rates, dtype=np.float64)
    else:
        r_grid = np.array([_one(I) for I in I_grid], dtype=np.float64)
    return I_grid, r_grid


def fit_fi(
    model: str,
    type_id: str = "default",
    params: dict[str, float] | None = None,
    I_min: float = 0.0,
    I_max: float = 1.0,
    n_I: int = 32,
    t_total: float = 1500.0,
    window: float = 800.0,
    dt: float = 0.05,
    n_jobs: int = 1,
    n_validate: int = 8,
    chirp: bool = False,
) -> FittedActivation:
    I_grid, r_grid = scan_fi_curve(
        model,
        params=params,
        I_min=I_min,
        I_max=I_max,
        n_I=n_I,
        t_total=t_total,
        window=window,
        dt=dt,
        n_jobs=n_jobs,
    )
    notes = []
    if _spline_overshoots(I_grid, r_grid):
        method = "pchip"
        notes.append("CubicSpline overshoot or Type-II gap; using PCHIP")
    else:
        method = "pchip" if model == "hh" else "linear"
        if model == "hh":
            notes.append("HH uses PCHIP to respect onset gap / depolarization block")
    I_onset = _onset_current(I_grid, r_grid)
    interp = PchipInterpolator(I_grid, r_grid, extrapolate=False)
    rng = np.random.default_rng(0)
    I_val = rng.uniform(I_min, I_max, size=n_validate)
    r_true = np.array(
        [
            firing_rate(model, params, float(I), t_total=t_total, dt=dt, window=window)
            for I in I_val
        ]
    )
    r_pred = np.array(interp(np.clip(I_val, I_grid[0], I_grid[-1])))
    r_pred = np.where(I_val < I_onset, 0.0, r_pred)
    r_pred = np.clip(r_pred, 0.0, float(np.max(r_grid)) if r_grid.size else 0.0)
    r2 = _r2(r_true, r_pred)
    mse = float(np.mean((r_true - r_pred) ** 2))
    quality = "ok" if r2 >= R2_GATE else "poor"
    if quality == "poor":
        notes.append(f"R2={r2:.3f} below gate {R2_GATE}")
    z = None
    if chirp:
        z = chirp_impedance(model, params, I0=max(I_min, 0.02), amp=0.01)
        notes.append(z.notes)
    layer, decision = decide_layer(model, z, has_lut=False)
    return FittedActivation(
        schema_version=NPZ_SCHEMA_VERSION,
        type_id=type_id,
        model=model,
        params=dict(params or {}),
        method=method,
        I_grid=I_grid,
        r_grid=r_grid,
        I_onset=I_onset,
        r2=r2,
        mse=mse,
        quality=quality,
        notes="; ".join(notes),
        layer=layer,
        level_decision=decision,
        z_freqs=None if z is None else z.freqs_hz,
        z_abs=None if z is None else z.z_abs,
    )


def fit_spike_lut(
    model: str,
    params: dict[str, float] | None = None,
    I_min: float = 0.0,
    I_max: float = 1.0,
    n_I: int = 24,
    n_dt: int = 16,
    dt: float = 1.0,
    t_ref_max: float | None = None,
) -> dict[str, np.ndarray | float]:
    spec, p = resolve(model, params)
    t_ref = float(getattr(p, "t_ref", 2.0))
    t_span = float(t_ref_max if t_ref_max is not None else max(t_ref * 15.0, 40.0))
    I_axis = np.linspace(I_min, I_max, n_I)
    dt_axis = np.linspace(0.0, t_span, n_dt)
    spike_mask = np.zeros((n_I, n_dt), dtype=np.float32)
    V_next = np.zeros((n_I, n_dt), dtype=np.float32)
    vth = spec.spike_threshold(p)
    hybrid = spec.hybrid_reset()
    hits = 0
    total = 0
    sub = min(dt, 0.1)
    for i, I_val in enumerate(I_axis):
        for j, t_since in enumerate(dt_axis):
            y = spec.y0(p).astype(np.float64)
            if hybrid:
                y = spec.reset(y, p)
            elapsed = 0.0
            while elapsed + 1e-12 < t_since:
                in_ref = hybrid and elapsed < t_ref
                I_now = 0.0 if in_ref else float(I_val)
                y = y + sub * spec.rhs(0.0, y, I_now, p)
                if in_ref:
                    y[0] = float(p.Vreset)
                elapsed += sub
            v_prev = float(y[0])
            in_ref_now = hybrid and t_since < t_ref
            y = y + dt * spec.rhs(0.0, y, 0.0 if in_ref_now else float(I_val), p)
            crossed = v_prev < vth <= float(y[0])
            already_high = (not in_ref_now) and v_prev >= vth
            spiked = (not in_ref_now) and (crossed or already_high)
            if spiked and hybrid:
                y = spec.reset(y, p)
            elif in_ref_now and hybrid:
                y[0] = float(p.Vreset)
            spike_mask[i, j] = 1.0 if spiked else 0.0
            V_next[i, j] = float(y[0])
            total += 1
            hits += int(spiked)
    hit_rate = hits / max(total, 1)
    return {
        "lut_I": I_axis,
        "lut_dt": dt_axis,
        "lut_spike": spike_mask,
        "lut_V": V_next,
        "lut_hit_rate": float(hit_rate),
        "quality": "ok" if hit_rate > 0.0 else "poor",
    }


def attach_lut(fit: FittedActivation, lut: dict[str, Any]) -> FittedActivation:
    fit.lut_I = np.asarray(lut["lut_I"])
    fit.lut_dt = np.asarray(lut["lut_dt"])
    fit.lut_spike = np.asarray(lut["lut_spike"])
    fit.lut_V = np.asarray(lut["lut_V"])
    fit.lut_hit_rate = float(lut["lut_hit_rate"])
    if fit.layer == "L0":
        fit.layer = "L1"
        extra = "2D LUT attached"
        fit.level_decision = f"{fit.level_decision}; {extra}" if fit.level_decision else extra
    return fit


def save_fit(fit: FittedActivation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": np.array([fit.schema_version]),
        "I_grid": fit.I_grid,
        "r_grid": fit.r_grid,
        "I_onset": np.array([fit.I_onset]),
        "r2": np.array([fit.r2]),
        "mse": np.array([fit.mse]),
        "layer": np.array([fit.layer]),
        "level_decision": np.array([fit.level_decision]),
    }
    if fit.lut_I is not None:
        payload["lut_I"] = fit.lut_I
        payload["lut_dt"] = fit.lut_dt
        payload["lut_spike"] = fit.lut_spike
        payload["lut_V"] = fit.lut_V
    if fit.z_freqs is not None:
        payload["z_freqs"] = fit.z_freqs
        payload["z_abs"] = fit.z_abs
    if fit.srm is not None:
        payload["kappa_amps"] = fit.srm.kappa.amps
        payload["kappa_taus"] = fit.srm.kappa.taus
        payload["eta_amps"] = fit.srm.eta.amps
        payload["eta_taus"] = fit.srm.eta.taus
        payload["theta_amps"] = fit.srm.theta.amps
        payload["theta_taus"] = fit.srm.theta.taus
        payload["srm_vrh"] = np.array([fit.srm.v_rh])
    np.savez_compressed(path, **payload)
    meta = {
        "schema_version": fit.schema_version,
        "type_id": fit.type_id,
        "model": fit.model,
        "params": fit.params,
        "method": fit.method,
        "quality": fit.quality,
        "r2": fit.r2,
        "mse": fit.mse,
        "notes": fit.notes,
        "layer": fit.layer,
        "level_decision": fit.level_decision,
        "lut_hit_rate": fit.lut_hit_rate,
        "pysr_expr": fit.pysr_expr,
        "fallback_ode": fit.fallback_ode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = path.with_suffix(".meta.json")
    if path.suffix == ".npz":
        meta_path = Path(str(path)[:-4] + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_fit(path: str | Path) -> FittedActivation:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    schema = int(np.asarray(data["schema_version"]).reshape(-1)[0])
    if schema < NPZ_SCHEMA_VERSION - 1:
        raise ValueError(f"NPZ schema {schema} is too old (need >= {NPZ_SCHEMA_VERSION - 1})")
    meta_path = Path(str(path)[:-4] + ".meta.json") if path.suffix == ".npz" else path.with_suffix(".meta.json")
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    layer = str(np.asarray(data["layer"]).reshape(-1)[0]) if "layer" in data.files else str(meta.get("layer", "L0"))
    level_decision = (
        str(np.asarray(data["level_decision"]).reshape(-1)[0])
        if "level_decision" in data.files
        else str(meta.get("level_decision", ""))
    )
    lut_I = data["lut_I"] if "lut_I" in data.files else None
    srm = None
    if "kappa_amps" in data.files:
        srm = SRMKernels(
            kappa=ExpKernel(np.asarray(data["kappa_amps"]), np.asarray(data["kappa_taus"])),
            eta=ExpKernel(np.asarray(data["eta_amps"]), np.asarray(data["eta_taus"])),
            theta=ExpKernel(np.asarray(data["theta_amps"]), np.asarray(data["theta_taus"])),
            v_rh=float(np.asarray(data["srm_vrh"]).reshape(-1)[0]) if "srm_vrh" in data.files else -50.0,
        )
    return FittedActivation(
        schema_version=schema,
        type_id=str(meta.get("type_id", "unknown")),
        model=str(meta.get("model", "lif")),
        params={k: float(v) for k, v in dict(meta.get("params") or {}).items()},
        method=str(meta.get("method", "linear")),
        I_grid=np.asarray(data["I_grid"], dtype=np.float64),
        r_grid=np.asarray(data["r_grid"], dtype=np.float64),
        I_onset=float(np.asarray(data["I_onset"]).reshape(-1)[0]),
        r2=float(np.asarray(data["r2"]).reshape(-1)[0]),
        mse=float(np.asarray(data["mse"]).reshape(-1)[0]),
        quality=str(meta.get("quality", "unknown")),
        notes=str(meta.get("notes", "")),
        layer=layer,
        level_decision=level_decision,
        lut_I=None if lut_I is None else np.asarray(lut_I),
        lut_dt=None if "lut_dt" not in data.files else np.asarray(data["lut_dt"]),
        lut_spike=None if "lut_spike" not in data.files else np.asarray(data["lut_spike"]),
        lut_V=None if "lut_V" not in data.files else np.asarray(data["lut_V"]),
        lut_hit_rate=meta.get("lut_hit_rate"),
        srm=srm,
        z_freqs=None if "z_freqs" not in data.files else np.asarray(data["z_freqs"]),
        z_abs=None if "z_abs" not in data.files else np.asarray(data["z_abs"]),
        pysr_expr=meta.get("pysr_expr"),
        fallback_ode=bool(meta.get("fallback_ode", False)),
    )
