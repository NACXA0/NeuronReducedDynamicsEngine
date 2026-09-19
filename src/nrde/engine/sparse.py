"""Sparse synaptic current aggregation (CSR, cached on GraphData)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from nrde.progress import get_progress, iter_progress
from nrde.types import GraphData


def _csr_token(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int) -> tuple[int, ...]:
    return (
        id(edge_index),
        id(edge_weight),
        int(n_nodes),
        int(np.asarray(edge_weight).size),
    )


def _type_token(node_type: np.ndarray, n_types: int, n_nodes: int) -> tuple[int, ...]:
    return (id(node_type), int(n_types), int(n_nodes))


def _group_bounds(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if keys.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    if keys.size == 1:
        return np.zeros(1, dtype=np.int64), np.ones(1, dtype=np.int64)
    change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    starts = np.empty(change.size + 1, dtype=np.int64)
    ends = np.empty(change.size + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = change
    ends[:-1] = change
    ends[-1] = keys.size
    return starts, ends


def _place_edges(
    indices: np.ndarray,
    data: np.ndarray,
    cursor: np.ndarray,
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
) -> None:
    if pre.size == 0:
        return
    order = np.argsort(post, kind="stable")
    post_s = post[order]
    starts, ends = _group_bounds(post_s)
    rows = post_s[starts]
    dest = cursor[rows]
    lengths = ends - starts
    cursor[rows] = dest + lengths
    pos = np.repeat(dest, lengths)
    pos += np.arange(post_s.size, dtype=np.int64) - np.repeat(starts, lengths)
    indices[pos] = pre[order]
    data[pos] = weight[order]


def _build_csr_chunked(
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    n_nodes: int,
    chunk: int,
) -> csr_matrix:
    n_edges = int(pre.shape[0])
    counts = np.bincount(post, minlength=n_nodes)
    indptr = np.empty(n_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(n_edges, dtype=np.int64)
    data = np.empty(n_edges, dtype=np.float32)
    cursor = indptr[:-1].copy()
    for start, end in iter_progress(n_edges, "building CSR", "edges", chunk):
        _place_edges(indices, data, cursor, pre[start:end], post[start:end], weight[start:end])
    mat = csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes), dtype=np.float32)
    # cuSPARSE wants columns sorted and duplicate (post, pre) pairs summed.
    mat.sum_duplicates()
    return mat


def build_csr(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int) -> csr_matrix:
    """I_syn = A @ x with A[post, pre] = w (duplicate edges are summed)."""
    n_nodes = int(n_nodes)
    if n_nodes <= 0:
        return csr_matrix((0, 0), dtype=np.float32)
    pre_post = np.asarray(edge_index)
    w = np.ascontiguousarray(edge_weight, dtype=np.float32)
    if pre_post.size == 0 or w.size == 0:
        return csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    pre = np.asarray(pre_post[0], dtype=np.int64)
    post = np.asarray(pre_post[1], dtype=np.int64)
    progress = get_progress()
    if progress.enabled and pre.size > progress.chunk:
        return _build_csr_chunked(pre, post, w, n_nodes, progress.chunk)
    return csr_matrix((w, (post, pre)), shape=(n_nodes, n_nodes), dtype=np.float32)


def indices_for_codes(codes: np.ndarray, n_types: int) -> list[np.ndarray]:
    """Node indices for each type code in ``0 .. n_types-1``.

    One stable argsort over the code array. A thread per type would rescan the
    same buffer ``n_types`` times; that pass is memory-bandwidth bound, so extra
    cores do not help and the scans dominate.
    """
    n_types = int(n_types)
    if n_types <= 0:
        return []
    codes = np.asarray(codes).reshape(-1)
    if codes.size == 0:
        return [np.empty(0, dtype=np.intp) for _ in range(n_types)]
    if not np.issubdtype(codes.dtype, np.integer):
        codes = codes.astype(np.int64, copy=False)
    lo = int(codes.min())
    hi = int(codes.max())
    if lo >= 0 and hi < n_types:
        return _split_in_range(codes, n_types)
    valid = (codes >= 0) & (codes < n_types)
    picked = np.flatnonzero(valid)
    parts = _split_in_range(np.asarray(codes[valid]), n_types)
    return [picked[part] for part in parts]


def _split_fast(codes: np.ndarray, n_types: int) -> list[np.ndarray]:
    if codes.size == 0:
        return [np.empty(0, dtype=np.intp) for _ in range(n_types)]
    if n_types == 1:
        return [np.arange(codes.size, dtype=np.intp)]
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_types)
    start = 0
    out: list[np.ndarray] = []
    for end in np.cumsum(counts).tolist():
        out.append(order[start:end])
        start = int(end)
    return out


def _place_codes(order: np.ndarray, cursor: np.ndarray, codes: np.ndarray, base: int) -> None:
    if codes.size == 0:
        return
    local = np.argsort(codes, kind="stable")
    sorted_codes = codes[local]
    starts, ends = _group_bounds(sorted_codes)
    rows = sorted_codes[starts]
    dest = cursor[rows]
    lengths = ends - starts
    cursor[rows] = dest + lengths
    pos = np.repeat(dest, lengths)
    pos += np.arange(sorted_codes.size, dtype=np.int64) - np.repeat(starts, lengths)
    order[pos] = base + local


def _split_chunked(codes: np.ndarray, n_types: int, chunk: int) -> list[np.ndarray]:
    counts = np.bincount(codes, minlength=n_types)
    starts_at = np.empty(n_types, dtype=np.int64)
    starts_at[0] = 0
    if n_types > 1:
        np.cumsum(counts[:-1], out=starts_at[1:])
    order = np.empty(codes.size, dtype=np.intp)
    cursor = starts_at.copy()
    label = f"indexing types ({n_types} types)"
    for start, end in iter_progress(int(codes.size), label, "nodes", chunk):
        _place_codes(order, cursor, codes[start:end], start)
    start = 0
    out: list[np.ndarray] = []
    for end in np.cumsum(counts).tolist():
        out.append(order[start:end])
        start = int(end)
    return out


def _split_in_range(codes: np.ndarray, n_types: int) -> list[np.ndarray]:
    progress = get_progress()
    if progress.enabled and codes.size > progress.chunk:
        return _split_chunked(codes, n_types, progress.chunk)
    return _split_fast(codes, n_types)


def invalidate_graph_cache(graph: GraphData) -> None:
    graph._csr = None
    graph._csr_token = None
    graph._type_indices = None
    graph._type_token = None
    graph._torch = None


def get_csr(graph: GraphData) -> csr_matrix:
    token = _csr_token(graph.edge_index, graph.edge_weight, graph.n_nodes)
    if graph._csr is not None and graph._csr_token == token:
        return graph._csr
    mat = build_csr(graph.edge_index, graph.edge_weight, graph.n_nodes)
    graph._csr = mat
    graph._csr_token = token
    return mat


def get_type_indices(graph: GraphData, n_types: int) -> list[np.ndarray]:
    token = _type_token(graph.node_type, n_types, graph.n_nodes)
    if graph._type_indices is not None and graph._type_token == token:
        return graph._type_indices
    nt = np.asarray(graph.node_type)
    n_types = int(n_types)
    indices = indices_for_codes(nt, n_types)
    graph._type_indices = indices
    graph._type_token = token
    return indices


def prepare_graph(graph: GraphData, n_types: int | None = None) -> GraphData:
    """Warm CSR and per-type index caches for the online loop."""
    get_csr(graph)
    if n_types is None:
        n_types = int(np.max(graph.node_type) + 1) if graph.n_nodes else 0
    if n_types:
        get_type_indices(graph, n_types)
    return graph


def sparse_matvec(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    x: np.ndarray,
    n_nodes: int,
    *,
    csr: csr_matrix | None = None,
) -> np.ndarray:
    """I_syn[post] += w * x[pre]. Prefers a cached CSR; builds one otherwise."""
    n_nodes = int(n_nodes)
    if n_nodes <= 0:
        return np.zeros(0, dtype=np.float64)
    mat = csr if csr is not None else build_csr(edge_index, edge_weight, n_nodes)
    if mat.shape[0] == 0 or mat.nnz == 0:
        return np.zeros(n_nodes, dtype=np.float64)
    x_m = np.ascontiguousarray(x, dtype=mat.dtype).reshape(-1)
    out = mat @ x_m
    return np.asarray(out, dtype=np.float64)
