"""Rate-mode engine: r = F(W @ r + I_ext) (FR-4.1)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nrde.engine.backends import resolve_engine_backend
from nrde.engine.sparse import get_csr, get_type_indices, prepare_graph, sparse_matvec
from nrde.types import FittedActivation, GraphData, RateState


def batched_apply(
    tables: list[FittedActivation],
    I: np.ndarray,
    node_type: np.ndarray,
    type_indices: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    r = np.zeros_like(I, dtype=np.float64)
    oor = 0
    indices = type_indices
    if indices is None:
        nt = np.asarray(node_type)
        indices = [np.flatnonzero(nt == t) for t in range(len(tables))]
    for t, table in enumerate(tables):
        idx = indices[t]
        if idx.size == 0:
            continue
        rt, n = table.eval_rate(I[idx])
        r[idx] = rt
        oor += n
    return r, oor


def init_rate_state(n_nodes: int, r0: float = 0.0) -> RateState:
    return RateState(r=np.full(n_nodes, r0, dtype=np.float64), out_of_range=0)


def step_rate(
    state: RateState,
    graph: GraphData,
    tables: list[FittedActivation],
    I_ext: np.ndarray,
    dt: float = 1.0,
    type_indices: list[np.ndarray] | None = None,
) -> RateState:
    del dt
    csr = get_csr(graph)
    I_syn = sparse_matvec(graph.edge_index, graph.edge_weight, state.r, graph.n_nodes, csr=csr)
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    if type_indices is None:
        type_indices = get_type_indices(graph, len(tables))
    r_new, n_oor = batched_apply(tables, I_total, graph.node_type, type_indices=type_indices)
    return replace(state, r=r_new, out_of_range=state.out_of_range + n_oor)


def run_rate(
    graph: GraphData,
    tables: list[FittedActivation],
    n_steps: int,
    I_ext: np.ndarray | None = None,
    dt: float = 1.0,
    r0: float = 0.0,
    record_trace: bool = True,
    backend: str | None = None,
) -> tuple[RateState, np.ndarray]:
    engine = resolve_engine_backend(backend)
    if engine == "torch":
        from nrde.engine.torch_rate import run_rate_torch

        return run_rate_torch(
            graph,
            tables,
            n_steps,
            I_ext=I_ext,
            dt=dt,
            r0=r0,
            record_trace=record_trace,
        )
    I_ext = np.zeros(graph.n_nodes, dtype=np.float64) if I_ext is None else np.asarray(I_ext, dtype=np.float64)
    prepare_graph(graph, n_types=len(tables))
    type_indices = get_type_indices(graph, len(tables))
    state = init_rate_state(graph.n_nodes, r0=r0)
    trace = np.zeros((n_steps if record_trace else 0, graph.n_nodes), dtype=np.float64)
    for t in range(n_steps):
        state = step_rate(state, graph, tables, I_ext, dt=dt, type_indices=type_indices)
        if record_trace:
            trace[t] = state.r
    return state, trace
