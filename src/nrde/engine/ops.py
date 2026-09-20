"""Pluggable rate-eval pipeline. Default: type table, then place, then modulator.

Place and modulator are identity until a teacher fills them (RFC-003: extra
axes are switches, not the engine contract). Custom steps register by name.
They are not serialized into npz.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Protocol

import numpy as np

from nrde.place import apply_place_mod
from nrde.types import FittedActivation

DEFAULT_PIPELINE = ("type", "place", "modulator")


class RateOp(Protocol):
    name: str

    def is_identity(self, fit: FittedActivation) -> bool: ...

    def apply(
        self,
        r: np.ndarray,
        I: np.ndarray,
        fit: FittedActivation,
        region: np.ndarray | None,
        xyz: np.ndarray | None,
    ) -> tuple[np.ndarray, int]: ...


class TypeOp:
    name = "type"

    def is_identity(self, fit: FittedActivation) -> bool:
        del fit
        return False

    def apply(
        self,
        r: np.ndarray,
        I: np.ndarray,
        fit: FittedActivation,
        region: np.ndarray | None,
        xyz: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        del r, region, xyz
        return fit.eval_rate(I)


class PlaceOp:
    name = "place"

    def is_identity(self, fit: FittedActivation) -> bool:
        return fit.place.is_identity()

    def apply(
        self,
        r: np.ndarray,
        I: np.ndarray,
        fit: FittedActivation,
        region: np.ndarray | None,
        xyz: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        return apply_place_mod(r, I, fit, region, xyz)


class ModulatorOp:
    """Reserved slow-modulator slot. Identity until a teacher fills it."""

    name = "modulator"

    def is_identity(self, fit: FittedActivation) -> bool:
        del fit
        return True

    def apply(
        self,
        r: np.ndarray,
        I: np.ndarray,
        fit: FittedActivation,
        region: np.ndarray | None,
        xyz: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        del I, fit, region, xyz
        return r, 0


_BUILTINS: dict[str, RateOp] = {
    "type": TypeOp(),
    "place": PlaceOp(),
    "modulator": ModulatorOp(),
}
_OPS: dict[str, RateOp] = dict(_BUILTINS)
_PIPELINE: ContextVar[tuple[str, ...]] = ContextVar("nrde_rate_pipeline", default=DEFAULT_PIPELINE)


def _known() -> str:
    return ", ".join(sorted(_OPS)) or "(empty)"


def register_op(op: RateOp, *, overwrite: bool = False) -> RateOp:
    name = str(op.name)
    if not name:
        raise ValueError("RateOp.name must be non-empty")
    if name in _OPS and not overwrite:
        raise ValueError(f"Rate op {name!r} already registered. Known: {_known()}")
    _OPS[name] = op
    return op


def get_op(name: str) -> RateOp:
    try:
        return _OPS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown rate op {name!r}. Known: {_known()}") from exc


def list_ops() -> list[str]:
    return sorted(_OPS)


def get_pipeline() -> tuple[str, ...]:
    return _PIPELINE.get()


def set_pipeline(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    resolved = tuple(str(item) for item in names)
    if not resolved:
        raise ValueError("Rate pipeline must not be empty")
    for name in resolved:
        get_op(name)
    _PIPELINE.set(resolved)
    return resolved


def append_op(name: str) -> tuple[str, ...]:
    get_op(name)
    current = get_pipeline()
    if name in current:
        raise ValueError(f"Rate op {name!r} is already in the pipeline {list(current)}")
    return set_pipeline(current + (name,))


def replace_op(old: str, new: str) -> tuple[str, ...]:
    get_op(new)
    names = list(get_pipeline())
    try:
        idx = names.index(old)
    except ValueError as exc:
        raise ValueError(f"Rate op {old!r} is not in the pipeline {names}. Known: {_known()}") from exc
    names[idx] = new
    return set_pipeline(names)


def reset_pipeline() -> None:
    """Restore the default pipeline and drop user-registered ops."""
    _OPS.clear()
    _OPS.update(_BUILTINS)
    _PIPELINE.set(DEFAULT_PIPELINE)


def pipeline_allows_torch(tables: list[FittedActivation]) -> bool:
    """CUDA kernel only implements type eval. Extra or non-identity steps stay on NumPy."""
    if get_pipeline() != DEFAULT_PIPELINE:
        return False
    place = get_op("place")
    modulator = get_op("modulator")
    return all(place.is_identity(table) and modulator.is_identity(table) for table in tables)


def apply_pipeline(
    fit: FittedActivation,
    I: np.ndarray,
    region: np.ndarray | None,
    xyz: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    current = np.asarray(I, dtype=np.float64)
    r = np.zeros(current.shape[0], dtype=np.float64)
    oor = 0
    for name in get_pipeline():
        op = get_op(name)
        if name != "type" and op.is_identity(fit):
            continue
        r, extra = op.apply(r, current, fit, region, xyz)
        oor += extra
    return r, oor
