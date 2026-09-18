"""Bind fitted activations onto graph type names."""

from __future__ import annotations

from pathlib import Path

import yaml

from nrde.types import FittedActivation, GraphData, TypeEntry


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
        )
        for item in raw
    ]
