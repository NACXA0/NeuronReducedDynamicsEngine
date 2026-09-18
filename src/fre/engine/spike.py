"""Spike-mode engine: 2D LUT or Izhikevich 2D ODE fallback (FR-4.2, R3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fre.engine.sparse import sparse_matvec
from fre.sim import resolve
from fre.types import FittedActivation, GraphData, SpikeState


def _bilinear_lookup(
    I: np.ndarray,
    t_since: np.ndarray,
    I_axis: np.ndarray,
    dt_axis: np.ndarray,
    spike_tab: np.ndarray,
    V_tab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    I_clip = np.clip(I, I_axis[0], I_axis[-1])
    t_clip = np.clip(t_since, dt_axis[0], dt_axis[-1])
    oor = int(np.count_nonzero((I < I_axis[0]) | (I > I_axis[-1])))
    i = np.interp(I_clip, I_axis, np.arange(I_axis.size)).astype(np.float64)
    j = np.interp(t_clip, dt_axis, np.arange(dt_axis.size)).astype(np.float64)
    i0 = np.clip(np.floor(i).astype(int), 0, I_axis.size - 2)
    j0 = np.clip(np.floor(j).astype(int), 0, dt_axis.size - 2)
    di = i - i0
    dj = j - j0

    def blend(tab: np.ndarray) -> np.ndarray:
        return (
            tab[i0, j0] * (1 - di) * (1 - dj)
            + tab[i0 + 1, j0] * di * (1 - dj)
            + tab[i0, j0 + 1] * (1 - di) * dj
            + tab[i0 + 1, j0 + 1] * di * dj
        )

    V = blend(V_tab)
    i_nn = np.clip(np.round(i).astype(int), 0, I_axis.size - 1)
    j_nn = np.clip(np.round(j).astype(int), 0, dt_axis.size - 1)
    spikes = spike_tab[i_nn, j_nn] >= 0.5
    return spikes, V, oor


def init_spike_state(
    graph: GraphData,
    tables: list[FittedActivation],
    use_izhikevich: bool = False,
) -> SpikeState:
    n = graph.n_nodes
    V0 = np.full(n, -70.0, dtype=np.float64)
    for t, table in enumerate(tables):
        mask = graph.node_type == t
        if table.lut_V is not None:
            V0[mask] = float(np.mean(table.lut_V))
        elif table.model == "izhikevich":
            V0[mask] = -65.0
    u = np.zeros(n, dtype=np.float64) if use_izhikevich else None
    return SpikeState(
        V=V0,
        t_since=np.full(n, 10.0, dtype=np.float64),
        spikes=np.zeros(n, dtype=bool),
        I_syn=np.zeros(n, dtype=np.float64),
        u=u,
        out_of_range=0,
    )


def step_spike_lut(
    state: SpikeState,
    graph: GraphData,
    tables: list[FittedActivation],
    I_ext: np.ndarray,
    dt: float = 1.0,
    tau_syn: float = 2.0,
) -> SpikeState:
    decay = float(np.exp(-dt / tau_syn))
    pulse = sparse_matvec(
        graph.edge_index,
        graph.edge_weight,
        state.spikes.astype(np.float64),
        graph.n_nodes,
    )
    I_syn = state.I_syn * decay + pulse
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    spikes = np.zeros(graph.n_nodes, dtype=bool)
    V = state.V.copy()
    t_since = state.t_since + dt
    oor = 0
    for t, table in enumerate(tables):
        mask = graph.node_type == t
        if not np.any(mask):
            continue
        if table.lut_I is None or table.lut_spike is None or table.lut_V is None or table.lut_dt is None:
            raise ValueError(f"Type {table.type_id} has no spike LUT")
        sp, v_new, n_oor = _bilinear_lookup(
            I_total[mask],
            state.t_since[mask],
            table.lut_I,
            table.lut_dt,
            table.lut_spike,
            table.lut_V,
        )
        spikes[mask] = sp
        V[mask] = v_new
        oor += n_oor
        t_since[mask] = np.where(sp, 0.0, state.t_since[mask] + dt)
    return replace(
        state,
        V=V,
        t_since=t_since,
        spikes=spikes,
        I_syn=I_syn,
        out_of_range=state.out_of_range + oor,
    )


def step_spike_izhikevich(
    state: SpikeState,
    graph: GraphData,
    tables: list[FittedActivation],
    I_ext: np.ndarray,
    dt: float = 1.0,
    tau_syn: float = 2.0,
    substeps: int = 2,
) -> SpikeState:
    """R3 fallback: explicit 2D Izhikevich, never a 3D LUT."""
    if state.u is None:
        state = replace(state, u=np.zeros_like(state.V))
    decay = float(np.exp(-dt / tau_syn))
    pulse = sparse_matvec(
        graph.edge_index,
        graph.edge_weight,
        state.spikes.astype(np.float64),
        graph.n_nodes,
    )
    I_syn = state.I_syn * decay + pulse
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    v = state.V.copy()
    u = state.u.copy()
    spikes = np.zeros(graph.n_nodes, dtype=bool)
    h = dt / substeps
    for t, table in enumerate(tables):
        mask = graph.node_type == t
        if not np.any(mask):
            continue
        spec, p = resolve("izhikevich", table.params if table.model == "izhikevich" else {})
        vv = v[mask]
        uu = u[mask]
        I = I_total[mask]
        spiked = np.zeros(vv.shape[0], dtype=bool)
        for _ in range(substeps):
            dv = 0.04 * vv * vv + 5.0 * vv + 140.0 - uu + I
            du = p.a * (p.b * vv - uu)
            vv = vv + h * dv
            uu = uu + h * du
            hit = vv >= p.Vth
            if np.any(hit):
                spiked |= hit
                vv = np.where(hit, p.c, vv)
                uu = np.where(hit, uu + p.d, uu)
        v[mask] = vv
        u[mask] = uu
        spikes[mask] = spiked
    t_since = np.where(spikes, 0.0, state.t_since + dt)
    return replace(
        state,
        V=v,
        u=u,
        t_since=t_since,
        spikes=spikes,
        I_syn=I_syn,
    )


def run_spike(
    graph: GraphData,
    tables: list[FittedActivation],
    n_steps: int,
    I_ext: np.ndarray | None = None,
    dt: float = 1.0,
    mode: str = "lut",
) -> tuple[SpikeState, np.ndarray]:
    I_ext = np.zeros(graph.n_nodes) if I_ext is None else np.asarray(I_ext, dtype=np.float64)
    use_izh = mode == "izhikevich"
    state = init_spike_state(graph, tables, use_izhikevich=use_izh)
    trace = np.zeros((n_steps, graph.n_nodes), dtype=bool)
    step = step_spike_izhikevich if use_izh else step_spike_lut
    for t in range(n_steps):
        state = step(state, graph, tables, I_ext, dt=dt)
        trace[t] = state.spikes
    return state, trace
