"""Bind fitted activations onto graph type names."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from nrde.types import FittedActivation, GraphData, TeacherRef, TypeEntry


def tables_for_graph(
    graph: GraphData,
    by_name: dict[str, FittedActivation],
    default_key: str | None = None,
) -> list[FittedActivation]:
    tables: list[FittedActivation] = []
    fallback = by_name.get(default_key or "default excitatory")
    if fallback is None and by_name:
        fallback = next(iter(by_name.values()))
    for name in graph.type_names:
        if name in by_name:
            tables.append(by_name[name])
        elif fallback is not None:
            tables.append(fallback)
        else:
            raise KeyError(f"No fitted F for type {name!r}")
    return tables


def parse_fit_map(items: list[str] | tuple[str, ...] | None) -> dict[str, Path]:
    """Parse repeated ``TYPE=PATH`` specs. Duplicate types are an error."""
    out: dict[str, Path] = {}
    for raw in items or ():
        spec = str(raw).strip()
        if "=" not in spec:
            raise ValueError(f"fit map {spec!r} must be TYPE=PATH")
        key, value = spec.split("=", 1)
        type_name = key.strip()
        path = value.strip()
        if not type_name or not path:
            raise ValueError(f"fit map {spec!r} must be TYPE=PATH")
        if type_name in out:
            raise ValueError(f"duplicate fit map type {type_name!r}")
        out[type_name] = Path(path)
    return out


def parse_type_names(raw: str | None) -> tuple[str, ...] | None:
    """Comma-separated graph type names for the random-graph path."""
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("type names must be a comma-separated list with no empty entries")
    return tuple(parts)


def bind_tables(
    graph: GraphData,
    by_name: Mapping[str, FittedActivation],
    fallback: FittedActivation | None = None,
) -> list[FittedActivation]:
    """Bind by graph type name. Unmapped types use ``fallback``; else KeyError.

    Unlike ``tables_for_graph``, the first table is not an implicit fallback.
    """
    tables: list[FittedActivation] = []
    missing: list[str] = []
    for name in graph.type_names:
        fit = by_name.get(name, fallback)
        if fit is None:
            missing.append(str(name))
        else:
            tables.append(fit)
    if missing:
        raise KeyError(f"No fitted F for type(s) {missing}")
    return tables


def bound_type_ids(graph: GraphData, tables: list[FittedActivation]) -> dict[str, str]:
    """Map each graph type name to the bound table's ``type_id``."""
    if len(tables) != len(graph.type_names):
        raise ValueError(
            f"table count {len(tables)} does not match type_names {len(graph.type_names)}"
        )
    return {name: table.type_id for name, table in zip(graph.type_names, tables, strict=True)}


def load_type_map(path: str) -> list[TypeEntry]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [
        TypeEntry(
            type_id=str(item["type_id"]),
            model=str(item["model"]),
            params={k: float(v) for k, v in dict(item.get("params") or {}).items()},
            ref=item.get("ref"),
            tier=str(item.get("tier", "default")),
            reduction_benefit=str(item.get("reduction_benefit", "control")),
            teacher=_teacher_from_item(item),
        )
        for item in raw
    ]


def _teacher_from_item(item: dict) -> TeacherRef | None:
    raw = item.get("teacher")
    if not isinstance(raw, dict):
        return None
    return TeacherRef(
        kind=str(raw.get("kind", "builtin")),
        ref=str(raw.get("ref", item.get("model", ""))),
        notes=str(raw.get("notes", "") or ""),
    )
