"""Local online-ODE fallback for types on the inapproximable list (DD-4, R4)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nrde.engine.rate import batched_apply
from nrde.engine.sparse import sparse_matvec
from nrde.sim import euler_step, resolve
from nrde.types import FittedActivation, GraphData, RateState


def fallback_mask(graph: GraphData, type_ids: set[str] | None) -> np.ndarray:
    if graph.fallback_mask is not None:
        return graph.fallback_mask.astype(bool)
    if not type_ids:
        return np.zeros(graph.n_nodes, dtype=bool)
    names = np.array(graph.type_names)
    wanted = np.array([names[int(t)] in type_ids for t in graph.node_type])
    return wanted


def step_rate_mixed(
    state: RateState,
    graph: GraphData,
    tables: list[FittedActivation],
    I_ext: np.ndarray,
    dt: float = 1.0,
    fallback_types: set[str] | None = None,
) -> RateState:
    """Reduced F everywhere except listed types, which Euler-step their ODE."""
    I_syn = sparse_matvec(graph.edge_index, graph.edge_weight, state.r, graph.n_nodes)
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    r_new, n_oor = batched_apply(tables, I_total, graph.node_type)
    mask = fallback_mask(graph, fallback_types)
    if not np.any(mask):
        return replace(state, r=r_new, out_of_range=state.out_of_range + n_oor)
    y = state.y_ode
    if y is None:
        y = np.zeros((graph.n_nodes, 4), dtype=np.float64)
        for t, table in enumerate(tables):
            spec, p = resolve(table.model, table.params)
            m = (graph.node_type == t) & mask
            if np.any(m):
                y0 = spec.y0(p)
                y[m, : y0.size] = y0
    idx = np.where(mask)[0]
    spikes = np.zeros(graph.n_nodes, dtype=bool)
    for i in idx.tolist():
        table = tables[int(graph.node_type[i])]
        spec, p = resolve(table.model, table.params)
        n = spec.y0(p).size
        yi = y[i, :n]
        vth = spec.spike_threshold(p)
        yi = euler_step(spec, p, yi, float(I_total[i]), dt)
        if yi[0] >= vth:
            spikes[i] = True
            if spec.hybrid_reset():
                yi = spec.reset(yi, p)
        y[i, :n] = yi
        r_new[i] = (1000.0 / dt) if spikes[i] else 0.0
    return replace(state, r=r_new, y_ode=y, out_of_range=state.out_of_range + n_oor)
