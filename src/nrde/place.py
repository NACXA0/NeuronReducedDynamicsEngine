"""Region and xyz corrections on a type-bound F. Identity is a no-op."""

from __future__ import annotations

from typing import Any

import numpy as np

from nrde.types import FittedActivation, PlaceMod, TeacherRef


def apply_place_mod(
    r: np.ndarray,
    I: np.ndarray,
    fit: FittedActivation,
    region: np.ndarray | None,
    xyz: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Scale ``r`` and re-threshold onset. Returns the input array when identity.

    ``r *= gain * (1 + Δregion + xyz · c)``. NaN coordinates count as 0.
    Unknown region names contribute 0. Onset shift can only silence rates
    already produced by ``eval_rate``; it does not restore subthreshold points.
    """
    place = fit.place
    if place.is_identity():
        return r, 0
    r = np.array(r, dtype=np.float64, copy=True)
    current = np.asarray(I, dtype=np.float64)
    n = int(r.shape[0])
    region_gain = np.zeros(n, dtype=np.float64)
    region_onset = np.zeros(n, dtype=np.float64)
    if region is not None and (place.region_gain or place.region_onset):
        labels = np.asarray(region, dtype=str)
        uniq, inverse = np.unique(labels, return_inverse=True)
        gain_of = np.array([place.region_gain.get(str(name), 0.0) for name in uniq], dtype=np.float64)
        onset_of = np.array([place.region_onset.get(str(name), 0.0) for name in uniq], dtype=np.float64)
        region_gain = gain_of[inverse]
        region_onset = onset_of[inverse]
    xyz_gain = 0.0
    xyz_onset = 0.0
    if xyz is not None:
        pts = np.nan_to_num(np.asarray(xyz, dtype=np.float64), nan=0.0)
        if pts.shape != (n, 3):
            raise ValueError(f"xyz shape {pts.shape} does not match rates {(n, 3)}")
        xyz_gain = pts @ place.xyz_gain
        xyz_onset = pts @ place.xyz_onset
    factor = place.gain * (1.0 + region_gain + xyz_gain)
    r *= factor
    onset = float(fit.I_onset) + region_onset + xyz_onset
    r = np.where(current < onset, 0.0, r)
    return r, 0


def place_to_meta(place: PlaceMod) -> dict[str, Any] | None:
    if place.is_identity():
        return None
    return {
        "gain": place.gain,
        "region_gain": dict(place.region_gain),
        "xyz_gain": place.xyz_gain.tolist(),
        "region_onset": dict(place.region_onset),
        "xyz_onset": place.xyz_onset.tolist(),
    }


def place_from_meta(raw: Any) -> PlaceMod:
    if not raw:
        return PlaceMod()
    if not isinstance(raw, dict):
        raise ValueError("place meta must be an object")
    return PlaceMod(
        gain=float(raw.get("gain", 1.0)),
        region_gain=dict(raw.get("region_gain") or {}),
        xyz_gain=np.asarray(raw.get("xyz_gain", (0.0, 0.0, 0.0)), dtype=np.float64),
        region_onset=dict(raw.get("region_onset") or {}),
        xyz_onset=np.asarray(raw.get("xyz_onset", (0.0, 0.0, 0.0)), dtype=np.float64),
    )


def teacher_to_meta(teacher: TeacherRef | None) -> dict[str, str] | None:
    if teacher is None:
        return None
    return {"kind": teacher.kind, "ref": teacher.ref, "notes": teacher.notes}


def teacher_from_meta(raw: Any) -> TeacherRef | None:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("teacher meta must be an object")
    return TeacherRef(
        kind=str(raw.get("kind", "builtin")),
        ref=str(raw.get("ref", "")),
        notes=str(raw.get("notes", "") or ""),
    )
