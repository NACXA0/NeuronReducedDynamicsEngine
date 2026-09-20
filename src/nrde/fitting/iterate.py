"""Capped multi-round fit: densify the f-I grid, then L1, then L1+L2.

``--auto`` is the only caller. A one-shot ``fit`` does not enter this loop.
L3 is a ceiling note (``l3=deferred_v0.2``), not an FNO weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nrde.fitting.fit import R2_GATE, _onset_current, attach_lut, fit_fi, fit_spike_lut
from nrde.fitting.srm import attach_srm, fit_srm_kernels
from nrde.models.lif import LIFParams, analytic_rate
from nrde.types import FittedActivation
from nrde.validation import r2_score, validate_single

_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}
_N_I_CAP = 64
_EXTRA_GRIDS = 2
_TIMING_MODELS = frozenset({"adexp", "hh"})
_PREFER = ("efficiency", "neutral", "fidelity")
# Neutral upgrades a passing L1 only when VP is still above this comfort band
# and the next layer lowers it by at least the margin. SRM often does not move VP.
NEUTRAL_VP_COMFORT = 0.05
NEUTRAL_VP_MARGIN = 0.02


def normalize_prefer(prefer: str) -> str:
    key = str(prefer).strip().lower()
    if key not in _PREFER:
        raise ValueError(f"Unknown prefer {prefer!r}. Known: {list(_PREFER)}")
    return key


def _rank(layer: str) -> int:
    key = str(layer).upper()
    if key not in _RANK:
        raise ValueError(f"Unknown layer {layer!r}. Known: {sorted(_RANK)}")
    return _RANK[key]


def _record(
    fit: FittedActivation,
    n_I: int,
    action: str,
    relative_vp: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "n_I": int(n_I),
        "r2": float(fit.r2),
        "relative_vp": None if relative_vp is None else float(relative_vp),
        "action": action,
    }
    vp = "None" if row["relative_vp"] is None else f"{row['relative_vp']:.4f}"
    bit = f"n_I={row['n_I']} r2={row['r2']:.4f} vp={vp} action={action}"
    fit.notes = f"{fit.notes}; {bit}" if fit.notes else bit
    return row


def _lif_params(fit: FittedActivation) -> LIFParams:
    fields = LIFParams.__dataclass_fields__
    picked = {key: float(fit.params[key]) for key in fit.params if key in fields}
    return LIFParams(**picked)


def _analytic_match(fit: FittedActivation) -> float:
    params = _lif_params(fit)
    pred = np.array([analytic_rate(float(i), params) for i in fit.I_grid], dtype=np.float64)
    return r2_score(fit.r_grid, pred)


def _fill_analytic(
    fit: FittedActivation,
    n_I: int,
    i_min: float,
    i_max: float,
    *,
    t_total: float,
    window: float,
    dt: float,
    n_validate: int,
) -> FittedActivation:
    """Replace the grid with the closed form. Holdout stays a few ODE points, not a full scan."""
    from nrde.sim import firing_rate

    params = _lif_params(fit)
    grid = np.linspace(float(i_min), float(i_max), int(n_I))
    rates = np.array([analytic_rate(float(i), params) for i in grid], dtype=np.float64)
    fit.I_grid = grid
    fit.r_grid = rates
    fit.I_onset = _onset_current(grid, rates)
    fit.method = "analytic"
    fit.config_hash = None
    rng = np.random.default_rng(0)
    probed = rng.uniform(float(i_min), float(i_max), size=max(int(n_validate), 2))
    r_true = np.array(
        [
            firing_rate("lif", fit.params or None, float(i), t_total=t_total, dt=dt, window=window)
            for i in probed
        ],
        dtype=np.float64,
    )
    r_pred = np.array([analytic_rate(float(i), params) for i in probed], dtype=np.float64)
    onset = float(fit.I_onset)
    r_pred = np.where(probed < onset, 0.0, r_pred)
    fit.r2 = r2_score(r_true, r_pred)
    fit.mse = float(np.mean((r_true - r_pred) ** 2))
    return fit


def _fit_rate(
    model: str,
    *,
    type_id: str,
    params: dict[str, float] | None,
    i_min: float,
    i_max: float,
    n_I: int,
    t_total: float,
    window: float,
    dt: float,
    n_validate: int,
    chirp: bool,
) -> FittedActivation:
    return fit_fi(
        model,
        type_id=type_id,
        params=params,
        I_min=i_min,
        I_max=i_max,
        n_I=n_I,
        t_total=t_total,
        window=window,
        dt=dt,
        n_validate=n_validate,
        chirp=chirp,
    )


def _mark_failed(fit: FittedActivation, max_layer: str, n_I: int) -> list[dict[str, Any]]:
    fit.quality = "poor"
    fit.fallback_ode = True
    rows: list[dict[str, Any]] = []
    if _rank(max_layer) >= _RANK["L3"]:
        bit = "l3=deferred_v0.2"
        fit.notes = f"{fit.notes}; {bit}" if fit.notes else bit
        rows.append(
            {
                "n_I": int(n_I),
                "r2": float(fit.r2),
                "relative_vp": None,
                "action": "l3_deferred",
            }
        )
    return rows


def fit_layers(
    layers: str,
    model: str,
    *,
    type_id: str = "default",
    params: dict[str, float] | None = None,
    I_min: float = 0.0,
    I_max: float = 0.8,
    n_I: int = 16,
    t_total: float = 800.0,
    window: float = 400.0,
    dt: float = 0.05,
    n_validate: int = 8,
    chirp: bool = False,
) -> tuple[FittedActivation, list[dict[str, Any]]]:
    """One shot at a named mix. Does not densify and does not climb."""
    key = str(layers).upper().replace(" ", "")
    if key not in {"L0", "L1", "L2", "L1+L2"}:
        raise ValueError(f"Unknown --layers {layers!r}. Use L0, L1, L2, or L1+L2.")
    fit = _fit_rate(
        model,
        type_id=type_id,
        params=params,
        i_min=I_min,
        i_max=I_max,
        n_I=n_I,
        t_total=t_total,
        window=window,
        dt=dt,
        n_validate=n_validate,
        chirp=chirp,
    )
    rounds = [_record(fit, n_I, "fit_fi")]
    want_lut = key in {"L1", "L1+L2"}
    want_srm = key in {"L2", "L1+L2"}
    if want_lut:
        fit = attach_lut(fit, fit_spike_lut(model, params, I_min=I_min, I_max=I_max, n_I=max(8, n_I // 2)))
        rounds.append(_record(fit, n_I, "attach_lut"))
    if want_srm:
        fit = attach_srm(fit, fit_srm_kernels(model, params, t_kernel=40.0, dt=0.1))
        rounds.append(_record(fit, n_I, "attach_srm"))
    fit.fallback_ode = False
    return fit, rounds


def fit_iterative(
    model: str,
    *,
    type_id: str = "default",
    params: dict[str, float] | None = None,
    I_min: float = 0.0,
    I_max: float = 0.8,
    n_I: int = 16,
    t_total: float = 800.0,
    window: float = 400.0,
    dt: float = 0.05,
    n_validate: int = 8,
    chirp: bool = False,
    max_layer: str = "L2",
    check_spikes: bool = False,
    prefer: str = "efficiency",
) -> tuple[FittedActivation, list[dict[str, Any]]]:
    """Climb while gates fail, and stop at ``max_layer``.

    ``prefer`` picks among fits that already pass. ``efficiency`` (default)
    keeps the cheapest. ``neutral`` tries the next layer only when spike error
    is still uncomfortable and actually drops. ``fidelity`` builds up to the ceiling.
    """
    prefer = normalize_prefer(prefer)
    ceiling = _rank(max_layer)
    n_I = max(2, int(n_I))
    rounds: list[dict[str, Any]] = []
    use_analytic = False
    fit: FittedActivation | None = None
    extras = 0
    while True:
        if use_analytic and fit is not None:
            fit = _fill_analytic(
                fit,
                n_I,
                I_min,
                I_max,
                t_total=t_total,
                window=window,
                dt=dt,
                n_validate=n_validate,
            )
            action = "analytic_grid"
        else:
            fit = _fit_rate(
                model,
                type_id=type_id,
                params=params,
                i_min=I_min,
                i_max=I_max,
                n_I=n_I,
                t_total=t_total,
                window=window,
                dt=dt,
                n_validate=n_validate,
                chirp=chirp,
            )
            action = "densify" if extras else "fit_fi"
            if model == "lif" and _analytic_match(fit) >= R2_GATE:
                fit.method = "analytic"
                fit.config_hash = None
                use_analytic = True
        assert fit is not None
        rounds.append(_record(fit, n_I, action))
        if float(fit.r2) >= R2_GATE or extras >= _EXTRA_GRIDS or n_I >= _N_I_CAP:
            break
        n_I = min(_N_I_CAP, n_I * 2)
        extras += 1

    rate_ok = float(fit.r2) >= R2_GATE
    # Fidelity climbs to the ceiling even for LIF. Efficiency/neutral add spike
    # checks only for adapting models, unless the caller asked.
    timing = ceiling >= _RANK["L1"] and (
        prefer == "fidelity" or model in _TIMING_MODELS or check_spikes
    )
    timing_ok = True
    if timing:
        fit = attach_lut(
            fit,
            fit_spike_lut(model, params, I_min=I_min, I_max=I_max, n_I=max(8, n_I // 2)),
        )
        report = validate_single(fit)
        rounds.append(_record(fit, n_I, "attach_lut", report.relative_vp))
        timing_ok = bool(report.nfr5_pass)
        before_vp = report.relative_vp
        uncomfortable = (
            prefer == "neutral"
            and timing_ok
            and before_vp is not None
            and float(before_vp) > NEUTRAL_VP_COMFORT
        )
        need_srm = ceiling >= _RANK["L2"] and (not timing_ok or prefer == "fidelity" or uncomfortable)
        if need_srm:
            fit = attach_srm(fit, fit_srm_kernels(model, params, t_kernel=40.0, dt=0.1))
            report = validate_single(fit)
            rounds.append(_record(fit, n_I, "attach_srm", report.relative_vp))
            improved = (
                before_vp is not None
                and report.relative_vp is not None
                and float(before_vp) - float(report.relative_vp) >= NEUTRAL_VP_MARGIN
            )
            if uncomfortable and not improved:
                fit.srm = None
                fit.layer = "L1" if fit.lut_I is not None else "L0"
                rounds.append(_record(fit, n_I, "revert_srm", before_vp))
            else:
                timing_ok = bool(report.nfr5_pass)
    tag = f"prefer={prefer}"
    fit.notes = f"{fit.notes}; {tag}" if fit.notes else tag
    if rate_ok and timing_ok:
        fit.quality = "ok"
        fit.fallback_ode = False
    else:
        rounds.extend(_mark_failed(fit, max_layer, n_I))
        rounds.append(_record(fit, int(fit.I_grid.size), "fallback_ode"))
    return fit, rounds
