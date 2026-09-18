"""Sparse synaptic current aggregation (CSR, cached on GraphData)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

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
    return csr_matrix((w, (post, pre)), shape=(n_nodes, n_nodes), dtype=np.float32)


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
    indices = [np.flatnonzero(nt == t) for t in range(int(n_types))]
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
