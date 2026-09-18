"""Connectome parser: Feather primary, CSV/Parquet compatible (FR-1, DD-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from fre.types import SHIU_ALPHA_NA, GraphData

PRE_ALIASES = ("pre_id", "body_pre", "bodyId_pre", "pre", "source", "i")
POST_ALIASES = ("post_id", "body_post", "bodyId_post", "post", "target", "j")
WEIGHT_ALIASES = ("synapse_count", "weight", "n_synapses", "syn_count", "roiWeight", "n")
NT_ALIASES = ("nt_type", "nt", "neurotransmitter", "consensus_nt", "pred_nt", "top_nt")
ID_ALIASES = ("bodyId", "body_id", "node_id", "id", "root_id")
TYPE_ALIASES = ("cell_type", "type", "cellType", "instance", "type_id")
REGION_ALIASES = ("region", "roi", "superclass", "super_class", "neuropil")

DEFAULT_NT_SIGN: dict[str, float] = {
    "acetylcholine": 1.0,
    "ach": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "glut": -1.0,
    "histamine": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
    "unknown": 1.0,
    "": 1.0,
}


def _pick_column(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _read_table(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow"}:
        import pyarrow.feather as feather

        table = feather.read_table(path)
        return {name: table.column(name).to_numpy() for name in table.column_names}
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return {name: table.column(name).to_numpy() for name in table.column_names}
    if suffix in {".csv", ".txt"}:
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            cols: dict[str, list[Any]] = {name: [] for name in reader.fieldnames}
            for row in reader:
                for name in reader.fieldnames:
                    cols[name].append(row[name])
        return {k: np.asarray(v) for k, v in cols.items()}
    raise ValueError(f"Unsupported connectome format: {path.suffix}")


def _as_int(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def load_alpha_table(path: str | Path | None = None) -> dict[str, float]:
    if path is None:
        return {"default": SHIU_ALPHA_NA, **{f"sign:{k}": v for k, v in DEFAULT_NT_SIGN.items()}}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, float] = {"default": float(raw.get("default", SHIU_ALPHA_NA))}
    for k, v in dict(raw.get("nt_sign") or DEFAULT_NT_SIGN).items():
        out[f"sign:{str(k).lower()}"] = float(v)
    for k, v in dict(raw.get("overrides") or {}).items():
        out[str(k)] = float(v)
    return out


def _alpha_for(
    nt: str,
    pre_type: str,
    table: Mapping[str, float],
) -> float:
    key = f"{nt}|{pre_type}"
    if key in table:
        return float(table[key])
    base = float(table.get("default", SHIU_ALPHA_NA))
    sign = float(table.get(f"sign:{nt.lower()}", table.get("sign:unknown", 1.0)))
    return base * sign


def load_connectome(
    synapses_path: str | Path,
    annotations_path: str | Path | None = None,
    nt_path: str | Path | None = None,
    alpha_table: Mapping[str, float] | None = None,
    alpha_path: str | Path | None = None,
    include_types: Iterable[str] | None = None,
    include_regions: Iterable[str] | None = None,
    include_ids: Iterable[int] | None = None,
) -> GraphData:
    syn = _read_table(synapses_path)
    pre_col = _pick_column(syn, PRE_ALIASES)
    post_col = _pick_column(syn, POST_ALIASES)
    w_col = _pick_column(syn, WEIGHT_ALIASES)
    nt_col = _pick_column(syn, NT_ALIASES)
    if pre_col is None or post_col is None:
        raise ValueError(f"Cannot find pre/post columns in {list(syn)}")
    pre = _as_int(syn[pre_col])
    post = _as_int(syn[post_col])
    weight = _as_float(syn[w_col]) if w_col is not None else np.ones(pre.shape[0])
    nt_edge = (
        np.asarray(syn[nt_col]).astype(str)
        if nt_col is not None
        else np.full(pre.shape[0], "unknown", dtype=object)
    )

    node_ids = np.unique(np.concatenate([pre, post]))
    type_of: dict[int, str] = {int(i): "unknown" for i in node_ids}
    region_of: dict[int, str] = {int(i): "" for i in node_ids}

    if annotations_path is not None:
        ann = _read_table(annotations_path)
        id_col = _pick_column(ann, ID_ALIASES)
        type_col = _pick_column(ann, TYPE_ALIASES)
        region_col = _pick_column(ann, REGION_ALIASES)
        if id_col is None:
            raise ValueError(f"Cannot find id column in annotations {list(ann)}")
        ids = _as_int(ann[id_col])
        types = np.asarray(ann[type_col]).astype(str) if type_col else np.full(ids.size, "unknown")
        regions = np.asarray(ann[region_col]).astype(str) if region_col else np.full(ids.size, "")
        for i, t, r in zip(ids.tolist(), types.tolist(), regions.tolist()):
            type_of[int(i)] = str(t)
            region_of[int(i)] = str(r)

    if nt_path is not None:
        nt_tab = _read_table(nt_path)
        id_col = _pick_column(nt_tab, ID_ALIASES)
        nt_n_col = _pick_column(nt_tab, NT_ALIASES)
        if id_col is not None and nt_n_col is not None:
            by_id = {
                int(i): str(n)
                for i, n in zip(_as_int(nt_tab[id_col]).tolist(), np.asarray(nt_tab[nt_n_col]).astype(str))
            }
            nt_edge = np.array(
                [by_id.get(int(p), str(n)) for p, n in zip(pre.tolist(), nt_edge.tolist())],
                dtype=object,
            )

    keep_nodes = set(int(i) for i in node_ids.tolist())
    if include_ids is not None:
        keep_nodes &= set(int(i) for i in include_ids)
    if include_types is not None:
        allowed = set(include_types)
        keep_nodes = {i for i in keep_nodes if type_of.get(i, "unknown") in allowed}
    if include_regions is not None:
        allowed_r = set(include_regions)
        keep_nodes = {i for i in keep_nodes if region_of.get(i, "") in allowed_r}

    mask = np.array([int(a) in keep_nodes and int(b) in keep_nodes for a, b in zip(pre, post)])
    pre, post, weight, nt_edge = pre[mask], post[mask], weight[mask], nt_edge[mask]
    node_ids = np.unique(np.concatenate([pre, post])) if pre.size else np.array([], dtype=np.int64)
    index_of = {int(i): k for k, i in enumerate(node_ids.tolist())}

    type_names = tuple(sorted({type_of.get(int(i), "unknown") for i in node_ids.tolist()} or {"unknown"}))
    type_index = {name: k for k, name in enumerate(type_names)}
    node_type = np.array(
        [type_index[type_of.get(int(i), "unknown")] for i in node_ids.tolist()],
        dtype=np.int32,
    )
    region = np.array([region_of.get(int(i), "") for i in node_ids.tolist()], dtype=object)

    alphas = dict(alpha_table) if alpha_table is not None else load_alpha_table(alpha_path)
    pre_types = np.array(
        [type_of.get(int(i), "unknown") for i in pre.tolist()],
        dtype=object,
    )
    edge_weight = np.array(
        [
            float(w) * _alpha_for(str(nt), str(pt), alphas)
            for w, nt, pt in zip(weight.tolist(), nt_edge.tolist(), pre_types.tolist())
        ],
        dtype=np.float32,
    )
    edge_index = np.vstack(
        [
            np.array([index_of[int(i)] for i in pre.tolist()], dtype=np.int64),
            np.array([index_of[int(i)] for i in post.tolist()], dtype=np.int64),
        ]
    )
    return GraphData(
        n_nodes=int(node_ids.size),
        node_ids=node_ids.astype(np.int64),
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_type=node_type,
        type_names=type_names,
        nt_type=np.asarray(nt_edge).astype(str),
        region=region,
        meta={"synapses_path": str(synapses_path)},
    )


def erdos_renyi_graph(
    n_nodes: int,
    p: float = 0.05,
    n_types: int = 2,
    seed: int = 0,
    weight: float = 0.002,
    type_names: tuple[str, ...] | None = None,
) -> GraphData:
    rng = np.random.default_rng(seed)
    pre, post = np.where(rng.random((n_nodes, n_nodes)) < p)
    mask = pre != post
    pre, post = pre[mask], post[mask]
    names = type_names or tuple(f"type_{k}" for k in range(n_types))
    node_type = rng.integers(0, len(names), size=n_nodes, dtype=np.int32)
    signs = np.where(node_type[pre] == 0, 1.0, -1.0)
    return GraphData(
        n_nodes=n_nodes,
        node_ids=np.arange(n_nodes, dtype=np.int64),
        edge_index=np.vstack([pre.astype(np.int64), post.astype(np.int64)]),
        edge_weight=(signs * weight).astype(np.float32),
        node_type=node_type,
        type_names=names,
        nt_type=np.where(signs > 0, "acetylcholine", "gaba").astype(str),
        region=np.full(n_nodes, "synthetic", dtype=object),
        meta={"generator": "erdos_renyi", "p": p, "seed": seed},
    )
