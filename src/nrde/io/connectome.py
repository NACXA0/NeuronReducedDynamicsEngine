"""Connectome parser: Feather primary, CSV/Parquet compatible (FR-1, DD-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import yaml

from nrde.progress import get_progress, iter_progress
from nrde.types import SHIU_ALPHA_NA, GraphData

PRE_ALIASES = ("pre_id", "body_pre", "bodyId_pre", "pre", "source", "i")
POST_ALIASES = ("post_id", "body_post", "bodyId_post", "post", "target", "j")
WEIGHT_ALIASES = ("synapse_count", "weight", "n_synapses", "syn_count", "roiWeight", "n")
NT_ALIASES = (
    "nt_type",
    "nt",
    "neurotransmitter",
    "consensus_nt",
    "pred_nt",
    "predicted_nt",
    "top_nt",
)
ID_ALIASES = ("bodyId", "body_id", "node_id", "id", "root_id", "body")
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


def _column_names(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow"}:
        import pyarrow as pa

        with pa.memory_map(str(path), "r") as source:
            return list(pa.ipc.open_file(source).schema.names)
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    if suffix in {".csv", ".txt"}:
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"CSV has no header: {path}") from exc
        return [name.strip() for name in header]
    raise ValueError(f"Unsupported connectome format: {path.suffix}")


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.columns.empty:
        raise ValueError(f"CSV has no header: {path}")
    return {str(name): frame[name].to_numpy() for name in frame.columns}


def _read_columns(path: str | Path, names: list[str]) -> dict[str, np.ndarray]:
    """Read only ``names``. Feather/Parquet stay columnar; no full-table copy."""
    path = Path(path)
    if not names:
        return {}
    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow"}:
        import pyarrow.feather as feather

        table = feather.read_table(path, columns=names)
        return {name: table.column(name).to_numpy(zero_copy_only=False) for name in names}
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=names)
        return {name: table.column(name).to_numpy(zero_copy_only=False) for name in names}
    if suffix in {".csv", ".txt"}:
        full = _read_csv(path)
        return {name: full[name] for name in names}
    raise ValueError(f"Unsupported connectome format: {path.suffix}")


def _read_ipc_columns(path: Path, names: list[str], label: str, unit: str) -> dict[str, np.ndarray]:
    """Stream Feather/Arrow batches. The batch count is known before the body is read."""
    import pyarrow as pa

    progress = get_progress()
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        n_batches = int(reader.num_record_batches)
        if n_batches == 0 or not names:
            return {name: np.array([]) for name in names}
        first = reader.get_batch(0)
        stride = int(first.num_rows)
        total = stride * n_batches
        cols: dict[str, np.ndarray] = {}
        for name in names:
            sample = first.column(name).to_numpy(zero_copy_only=False)
            cols[name] = np.empty(total, dtype=sample.dtype)
            cols[name][: sample.shape[0]] = sample
        offset = stride
        progress.tick(0, max(total, 1), label, unit)
        progress.tick(min(offset, total), max(total, 1), label, unit)
        for index in range(1, n_batches):
            batch = reader.get_batch(index)
            n = int(batch.num_rows)
            end = offset + n
            if end > cols[names[0]].shape[0]:
                total = end
                for name in names:
                    grown = np.empty(total, dtype=cols[name].dtype)
                    grown[:offset] = cols[name][:offset]
                    cols[name] = grown
            for name in names:
                cols[name][offset:end] = batch.column(name).to_numpy(zero_copy_only=False)
            offset = end
            progress.tick(offset, max(total, offset), label, unit)
        if offset != cols[names[0]].shape[0]:
            for name in names:
                cols[name] = cols[name][:offset]
            progress.tick(offset, offset, label, unit)
        return cols


def _read_parquet_columns(path: Path, names: list[str], label: str, unit: str) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    progress = get_progress()
    pf = pq.ParquetFile(path)
    total = int(pf.metadata.num_rows)
    if total == 0 or not names:
        return {name: np.array([]) for name in names}
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    seen = 0
    progress.tick(0, total, label, unit)
    for batch in pf.iter_batches(columns=names, batch_size=65_536):
        n = int(batch.num_rows)
        for name in names:
            parts[name].append(batch.column(name).to_numpy(zero_copy_only=False))
        seen += n
        progress.tick(seen, total, label, unit)
    return {
        name: np.concatenate(chunks) if len(chunks) > 1 else chunks[0] for name, chunks in parts.items()
    }


def _read_showing(path: Path, names: list[str], label: str, unit: str) -> dict[str, np.ndarray]:
    progress = get_progress()
    if not progress.enabled:
        return _read_columns(path, names)
    suffix = path.suffix.lower()
    if suffix in {".feather", ".arrow"}:
        return _read_ipc_columns(path, names, label, unit)
    if suffix in {".parquet", ".pq"}:
        return _read_parquet_columns(path, names, label, unit)
    data = _read_columns(path, names)
    n = int(next(iter(data.values())).shape[0]) if data else 0
    if n:
        progress.tick(n, n, label, unit)
    return data


def _union_endpoints(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    progress = get_progress()
    if not progress.enabled or pre.size <= progress.chunk:
        return np.union1d(pre, post)
    parts: list[np.ndarray] = []
    total = int(pre.shape[0])
    seen = 0
    progress.tick(0, total, "indexing nodes", "edges")
    for arr in (pre, post):
        for start in range(0, int(arr.shape[0]), progress.chunk):
            end = min(start + progress.chunk, int(arr.shape[0]))
            parts.append(np.unique(arr[start:end]))
            seen += end - start
            progress.tick(min(total - 1, seen // 2), total, "indexing nodes", "edges")
    if total > 1:
        progress.tick(total - 1, total, "indexing nodes", "edges")
    merged = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
    progress.tick(total, total, "indexing nodes", "edges")
    return merged


def _assign_nodes(
    node_ids: np.ndarray,
    ann_ids: np.ndarray,
    ann_types: np.ndarray,
    ann_regions: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Match ``_labels_for`` + ``_factorize``, ticking once per node chunk."""
    progress = get_progress()
    n = int(node_ids.size)
    if not progress.enabled or n <= progress.chunk:
        type_labels = _labels_for(node_ids, ann_ids, ann_types, "unknown")
        region = _labels_for(node_ids, ann_ids, ann_regions, "")
        type_names, node_type = _factorize(type_labels)
        return type_names, node_type, region
    raw = np.unique(np.asarray(ann_types, dtype=str)) if ann_types.size else np.array([], dtype=str)
    names_list = [str(item) for item in raw]
    if "unknown" not in names_list:
        names_list.append("unknown")
    names = np.array(sorted(set(names_list)), dtype=object)
    unknown = int(np.searchsorted(names, "unknown"))
    ann_codes = (
        np.searchsorted(names, np.asarray(ann_types, dtype=object)).astype(np.int32)
        if ann_types.size
        else np.zeros(0, dtype=np.int32)
    )
    node_type = np.empty(n, dtype=np.int32)
    region = np.empty(n, dtype=object)
    used = np.zeros(names.size, dtype=bool)
    label = f"assigning types ({int(raw.size)} annotation types)"
    for start, end in iter_progress(n, label, "nodes", progress.chunk):
        query = node_ids[start:end]
        width = end - start
        codes = np.full(width, unknown, dtype=np.int32)
        reg = np.empty(width, dtype=object)
        reg.fill("")
        if ann_ids.size:
            pos = np.searchsorted(ann_ids, query)
            valid = pos < ann_ids.size
            pos_c = np.minimum(pos, max(ann_ids.size - 1, 0))
            hit = valid & (ann_ids[pos_c] == query)
            codes[hit] = ann_codes[pos_c[hit]]
            if ann_regions.size:
                reg[hit] = ann_regions[pos_c[hit]]
        node_type[start:end] = codes
        region[start:end] = reg
        used[np.unique(codes)] = True
    keep = np.flatnonzero(used)
    if keep.size != names.size:
        remap = np.full(names.size, -1, dtype=np.int32)
        remap[keep] = np.arange(keep.size, dtype=np.int32)
        node_type = remap[node_type]
        names = names[keep]
    return tuple(str(item) for item in names), node_type, region


def _scale_weights(weight: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    progress = get_progress()
    values = np.asarray(weight, dtype=np.float64)
    if not progress.enabled or values.size <= progress.chunk:
        return (values * scale).astype(np.float32)
    out = np.empty(values.shape[0], dtype=np.float32)
    factor = np.asarray(scale, dtype=np.float64)
    scalar = factor.ndim == 0 or factor.size == 1
    for start, end in iter_progress(int(values.shape[0]), "scaling edges", "edges", progress.chunk):
        piece = float(factor) if scalar else factor[start:end]
        out[start:end] = (values[start:end] * piece).astype(np.float32, copy=False)
    return out


def _pack_edges(node_ids: np.ndarray, pre_i: np.ndarray, post: np.ndarray) -> np.ndarray:
    progress = get_progress()
    n = int(pre_i.shape[0])
    edge_index = np.empty((2, n), dtype=np.int64)
    if not progress.enabled or n <= progress.chunk:
        edge_index[0] = pre_i
        edge_index[1] = np.searchsorted(node_ids, post)
        return edge_index
    for start, end in iter_progress(n, "packing edges", "edges", progress.chunk):
        edge_index[0, start:end] = pre_i[start:end]
        edge_index[1, start:end] = np.searchsorted(node_ids, post[start:end])
    return edge_index


def _searchsorted_edges(node_ids: np.ndarray, query: np.ndarray) -> np.ndarray:
    progress = get_progress()
    if not progress.enabled or query.size <= progress.chunk:
        return np.searchsorted(node_ids, query)
    out = np.empty(query.shape[0], dtype=np.intp)
    for start, end in iter_progress(int(query.shape[0]), "mapping edges", "edges", progress.chunk):
        out[start:end] = np.searchsorted(node_ids, query[start:end])
    return out


def _as_int(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.int64:
        return arr
    return arr.astype(np.int64, copy=False)


def _in_sorted(sorted_ids: np.ndarray, query: np.ndarray) -> np.ndarray:
    if sorted_ids.size == 0:
        return np.zeros(query.shape, dtype=bool)
    pos = np.searchsorted(sorted_ids, query)
    valid = pos < sorted_ids.size
    pos = np.minimum(pos, sorted_ids.size - 1)
    return valid & (sorted_ids[pos] == query)


def _last_wins_sorted(ids: np.ndarray, *columns: np.ndarray) -> tuple[np.ndarray, ...]:
    """Unique ids, sorted, keeping the last row (same as a dict comprehension)."""
    ids = _as_int(ids)
    if ids.size == 0:
        return (ids, *tuple(np.asarray(col)[:0] for col in columns))
    _uniq, first_rev = np.unique(ids[::-1], return_index=True)
    keep = ids.size - 1 - first_rev
    return (_uniq, *(np.asarray(col)[keep] for col in columns))


def _labels_for(
    query: np.ndarray,
    sorted_ids: np.ndarray,
    labels: np.ndarray,
    default: str,
) -> np.ndarray:
    out = np.empty(query.shape, dtype=object)
    out.fill(default)
    if query.size == 0 or sorted_ids.size == 0:
        return out
    pos = np.searchsorted(sorted_ids, query)
    valid = pos < sorted_ids.size
    pos = np.minimum(pos, sorted_ids.size - 1)
    hit = valid & (sorted_ids[pos] == query)
    out[hit] = labels[pos[hit]]
    return out


def _factorize(labels: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    if labels.size == 0:
        return ("unknown",), np.zeros(0, dtype=np.int32)
    import pandas as pd

    codes, uniques = pd.factorize(np.asarray(labels, dtype=object), sort=True)
    return tuple(str(item) for item in uniques), np.asarray(codes, dtype=np.int32)


def _codes_to_objects(codes: np.ndarray, names: list[str]) -> np.ndarray:
    """Index into shared strings. Avoids a ``<U`` array, which blows up at 1e8 edges."""
    if codes.size == 0:
        return np.zeros(0, dtype=object)
    return np.asarray(names, dtype=object)[np.asarray(codes)]


def _ids_where(ids: np.ndarray, labels: np.ndarray, allowed: set[str]) -> np.ndarray:
    if ids.size == 0:
        return ids
    keep = np.isin(np.asarray(labels, dtype=str), np.asarray(sorted(allowed), dtype=str))
    return np.asarray(ids)[keep]


def _encode_labels(labels: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Map strings to codes with ``numpy`` search. ``unknown`` is always a category."""
    text = np.asarray(labels, dtype=str)
    names = np.unique(text) if text.size else np.array([], dtype=str)
    insert = int(np.searchsorted(names, "unknown"))
    if insert == names.size or names[insert] != "unknown":
        names = np.insert(names, insert, "unknown")
    codes = np.searchsorted(names, text).astype(np.int32, copy=False)
    return [str(name) for name in names], codes


def _edge_scale(
    nt_codes: np.ndarray,
    nt_names: list[str],
    pre_type_codes: np.ndarray,
    type_names: tuple[str, ...],
    alphas: Mapping[str, float],
) -> np.ndarray | float:
    base = float(alphas.get("default", SHIU_ALPHA_NA))
    unknown_sign = float(alphas.get("sign:unknown", 1.0))
    per_nt = np.array(
        [base * float(alphas.get(f"sign:{name.lower()}", unknown_sign)) for name in nt_names],
        dtype=np.float64,
    )
    nt_index = {name: i for i, name in enumerate(nt_names)}
    type_index = {name: i for i, name in enumerate(type_names)}
    overrides: list[tuple[int, int, float]] = []
    for key, val in alphas.items():
        if not isinstance(key, str) or "|" not in key:
            continue
        nt_s, ty_s = key.split("|", 1)
        ni = nt_index.get(nt_s)
        ti = type_index.get(ty_s)
        if ni is None or ti is None:
            continue
        overrides.append((ni, ti, float(val)))
    if nt_codes.size == 0:
        return float(per_nt[0]) if per_nt.size else base
    if not overrides and (per_nt.size == 1 or int(nt_codes.min()) == int(nt_codes.max())):
        return float(per_nt[int(nt_codes[0])])
    scale = per_nt[nt_codes]
    for ni, ti, val in overrides:
        scale[(nt_codes == ni) & (pre_type_codes == ti)] = val
    return scale


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
    """Load a connectome without building a Python object per edge.

    Filters and id remaps use sorted arrays and ``searchsorted``. Neurotransmitter
    labels stay as integer codes until the final shared-string column.
    """
    syn_path = Path(synapses_path)
    syn_names = _column_names(syn_path)
    pre_col = _pick_column(syn_names, PRE_ALIASES)
    post_col = _pick_column(syn_names, POST_ALIASES)
    w_col = _pick_column(syn_names, WEIGHT_ALIASES)
    nt_col = _pick_column(syn_names, NT_ALIASES)
    if pre_col is None or post_col is None:
        raise ValueError(f"Cannot find pre/post columns in {syn_names}")
    wanted = [pre_col, post_col]
    if w_col is not None:
        wanted.append(w_col)
    if nt_col is not None:
        wanted.append(nt_col)
    syn = _read_showing(syn_path, wanted, "reading synapses", "edges")
    pre = _as_int(syn.pop(pre_col))
    post = _as_int(syn.pop(post_col))
    weight = np.asarray(syn.pop(w_col)) if w_col is not None else np.ones(pre.shape[0], dtype=np.float64)
    raw_nt = np.asarray(syn.pop(nt_col)).astype(str) if nt_col is not None else None
    del syn

    ann_ids = np.zeros(0, dtype=np.int64)
    ann_types = np.zeros(0, dtype=object)
    ann_regions = np.zeros(0, dtype=object)
    if annotations_path is not None:
        ann_path = Path(annotations_path)
        ann_names = _column_names(ann_path)
        id_col = _pick_column(ann_names, ID_ALIASES)
        type_col = _pick_column(ann_names, TYPE_ALIASES)
        region_col = _pick_column(ann_names, REGION_ALIASES)
        if id_col is None:
            raise ValueError(f"Cannot find id column in annotations {ann_names}")
        ann_wanted = [id_col]
        if type_col is not None:
            ann_wanted.append(type_col)
        if region_col is not None:
            ann_wanted.append(region_col)
        ann = _read_showing(ann_path, ann_wanted, "reading annotations", "bodies")
        ann_ids, ann_types, ann_regions = _last_wins_sorted(
            ann[id_col],
            np.asarray(ann[type_col]).astype(str) if type_col else np.full(len(ann[id_col]), "unknown"),
            np.asarray(ann[region_col]).astype(str) if region_col else np.full(len(ann[id_col]), ""),
        )
        del ann

    mask = np.ones(pre.shape[0], dtype=bool)
    filtered = False
    if include_ids is not None:
        keep_ids = np.unique(np.asarray(list(include_ids), dtype=np.int64))
        mask &= _in_sorted(keep_ids, pre) & _in_sorted(keep_ids, post)
        filtered = True
    if include_types is not None:
        allowed = {str(name) for name in include_types}
        allowed_ids = _ids_where(ann_ids, ann_types, allowed)
        unknown_ok = "unknown" in allowed
        pre_ok = _in_sorted(allowed_ids, pre)
        post_ok = _in_sorted(allowed_ids, post)
        if unknown_ok:
            pre_ok = pre_ok | ~_in_sorted(ann_ids, pre)
            post_ok = post_ok | ~_in_sorted(ann_ids, post)
        mask &= pre_ok & post_ok
        filtered = True
    if include_regions is not None:
        allowed_r = {str(name) for name in include_regions}
        allowed_ids = _ids_where(ann_ids, ann_regions, allowed_r)
        unknown_ok = "" in allowed_r
        pre_ok = _in_sorted(allowed_ids, pre)
        post_ok = _in_sorted(allowed_ids, post)
        if unknown_ok:
            pre_ok = pre_ok | ~_in_sorted(ann_ids, pre)
            post_ok = post_ok | ~_in_sorted(ann_ids, post)
        mask &= pre_ok & post_ok
        filtered = True
    if filtered and not mask.all():
        pre, post, weight = pre[mask], post[mask], weight[mask]
        if raw_nt is not None:
            raw_nt = raw_nt[mask]
    del mask

    if pre.size == 0:
        return GraphData(
            n_nodes=0,
            node_ids=np.zeros(0, dtype=np.int64),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_weight=np.zeros(0, dtype=np.float32),
            node_type=np.zeros(0, dtype=np.int32),
            type_names=("unknown",),
            nt_type=np.zeros(0, dtype=object),
            region=np.zeros(0, dtype=object),
            meta={"synapses_path": str(syn_path)},
        )

    node_ids = _union_endpoints(pre, post)
    pre_i = _searchsorted_edges(node_ids, pre)
    type_names, node_type, region = _assign_nodes(node_ids, ann_ids, ann_types, ann_regions)
    if raw_nt is not None:
        nt_names_arr, nt_codes = np.unique(raw_nt, return_inverse=True)
        nt_names = [str(name) for name in nt_names_arr]
        nt_codes = nt_codes.astype(np.int32, copy=False)
        del raw_nt
    elif nt_path is not None:
        nt_file = Path(nt_path)
        nt_names_hdr = _column_names(nt_file)
        id_col = _pick_column(nt_names_hdr, ID_ALIASES)
        nt_n_col = _pick_column(nt_names_hdr, NT_ALIASES)
        if id_col is None or nt_n_col is None:
            nt_names = ["unknown"]
            nt_codes = np.zeros(pre.shape[0], dtype=np.int32)
        else:
            nt_tab = _read_showing(nt_file, [id_col, nt_n_col], "reading neurotransmitters", "bodies")
            nt_ids, nt_labels = _last_wins_sorted(nt_tab[id_col], np.asarray(nt_tab[nt_n_col]))
            del nt_tab
            nt_names, small_codes = _encode_labels(nt_labels)
            nt_codes = np.full(pre.shape[0], nt_names.index("unknown"), dtype=np.int32)
            pos = np.searchsorted(nt_ids, pre)
            valid = pos < nt_ids.size
            pos_c = np.minimum(pos, max(nt_ids.size - 1, 0))
            hit = valid & (nt_ids[pos_c] == pre)
            nt_codes[hit] = small_codes[pos_c[hit]]
            del nt_ids, nt_labels, small_codes
    else:
        nt_names = ["unknown"]
        nt_codes = np.zeros(pre.shape[0], dtype=np.int32)

    alphas = dict(alpha_table) if alpha_table is not None else load_alpha_table(alpha_path)
    scale = _edge_scale(nt_codes, nt_names, node_type[pre_i], type_names, alphas)
    edge_weight = _scale_weights(weight, scale)
    del weight, scale

    edge_index = _pack_edges(node_ids, pre_i, post)
    del pre, post, pre_i

    return GraphData(
        n_nodes=int(node_ids.size),
        node_ids=node_ids,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_type=node_type,
        type_names=type_names,
        nt_type=_codes_to_objects(nt_codes, nt_names),
        region=region,
        meta={"synapses_path": str(syn_path)},
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
