"""Rate-mode engine: r = F(W @ r + I_ext) (FR-4.1)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nrde.engine.sparse import sparse_matvec
from nrde.types import FittedActivation, GraphData, RateState


def batched_apply(
    tables: list[FittedActivation],
    I: np.ndarray,
    node_type: np.ndarray,
) -> tuple[np.ndarray, int]:
    r = np.zeros_like(I, dtype=np.float64)
    oor = 0
    for t, table in enumerate(tables):
        mask = node_type == t
        if not np.any(mask):
            continue
        rt, n = table.eval_rate(I[mask])
        r[mask] = rt
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
) -> RateState:
    del dt
    I_syn = sparse_matvec(graph.edge_index, graph.edge_weight, state.r, graph.n_nodes)
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    r_new, n_oor = batched_apply(tables, I_total, graph.node_type)
    return replace(state, r=r_new, out_of_range=state.out_of_range + n_oor)


def run_rate(
    graph: GraphData,
    tables: list[FittedActivation],
    n_steps: int,
    I_ext: np.ndarray | None = None,
    dt: float = 1.0,
    r0: float = 0.0,
) -> tuple[RateState, np.ndarray]:
    I_ext = np.zeros(graph.n_nodes) if I_ext is None else np.asarray(I_ext, dtype=np.float64)
    state = init_rate_state(graph.n_nodes, r0=r0)
    trace = np.zeros((n_steps, graph.n_nodes), dtype=np.float64)
    for t in range(n_steps):
        state = step_rate(state, graph, tables, I_ext, dt=dt)
        trace[t] = state.r
    return state, trace
