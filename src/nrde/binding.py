"""Bind fitted activations onto graph type names."""

from __future__ import annotations

from pathlib import Path

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
