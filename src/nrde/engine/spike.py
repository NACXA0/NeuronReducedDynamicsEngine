"""Spike-mode engine: 2D LUT or Izhikevich 2D ODE fallback (FR-4.2, R3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from nrde.engine.sparse import get_csr, get_type_indices, prepare_graph, sparse_matvec
from nrde.sim import resolve
from nrde.types import FittedActivation, GraphData, SpikeState


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
    indices = get_type_indices(graph, len(tables))
    for t, table in enumerate(tables):
        idx = indices[t]
        if idx.size == 0:
            continue
        if table.lut_V is not None:
            V0[idx] = float(np.mean(table.lut_V))
        elif table.model == "izhikevich":
            V0[idx] = -65.0
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
    type_indices: list[np.ndarray] | None = None,
) -> SpikeState:
    decay = float(np.exp(-dt / tau_syn))
    pulse = sparse_matvec(
        graph.edge_index,
        graph.edge_weight,
        state.spikes,
        graph.n_nodes,
        csr=get_csr(graph),
    )
    I_syn = state.I_syn * decay + pulse
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    spikes = np.zeros(graph.n_nodes, dtype=bool)
    V = state.V.copy()
    t_since = state.t_since + dt
    oor = 0
    indices = type_indices if type_indices is not None else get_type_indices(graph, len(tables))
    for t, table in enumerate(tables):
        idx = indices[t]
        if idx.size == 0:
            continue
        if table.lut_I is None or table.lut_spike is None or table.lut_V is None or table.lut_dt is None:
            raise ValueError(f"Type {table.type_id} has no spike LUT")
        sp, v_new, n_oor = _bilinear_lookup(
            I_total[idx],
            state.t_since[idx],
            table.lut_I,
            table.lut_dt,
            table.lut_spike,
            table.lut_V,
        )
        spikes[idx] = sp
        V[idx] = v_new
        oor += n_oor
        t_since[idx] = np.where(sp, 0.0, state.t_since[idx] + dt)
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
    type_indices: list[np.ndarray] | None = None,
) -> SpikeState:
    """R3 fallback: explicit 2D Izhikevich, never a 3D LUT."""
    if state.u is None:
        state = replace(state, u=np.zeros_like(state.V))
    decay = float(np.exp(-dt / tau_syn))
    pulse = sparse_matvec(
        graph.edge_index,
        graph.edge_weight,
        state.spikes,
        graph.n_nodes,
        csr=get_csr(graph),
    )
    I_syn = state.I_syn * decay + pulse
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    v = state.V.copy()
    u = state.u.copy()
    spikes = np.zeros(graph.n_nodes, dtype=bool)
    h = dt / substeps
    indices = type_indices if type_indices is not None else get_type_indices(graph, len(tables))
    for t, table in enumerate(tables):
        idx = indices[t]
        if idx.size == 0:
            continue
        _, p = resolve("izhikevich", table.params if table.model == "izhikevich" else {})
        vv = v[idx]
        uu = u[idx]
        I = I_total[idx]
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
        v[idx] = vv
        u[idx] = uu
        spikes[idx] = spiked
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
    record_trace: bool = True,
) -> tuple[SpikeState, np.ndarray]:
    I_ext = np.zeros(graph.n_nodes, dtype=np.float64) if I_ext is None else np.asarray(I_ext, dtype=np.float64)
    prepare_graph(graph, n_types=len(tables))
    type_indices = get_type_indices(graph, len(tables))
    use_izh = mode == "izhikevich"
    state = init_spike_state(graph, tables, use_izhikevich=use_izh)
    trace = np.zeros((n_steps if record_trace else 0, graph.n_nodes), dtype=bool)
    step = step_spike_izhikevich if use_izh else step_spike_lut
    for t in range(n_steps):
        state = step(state, graph, tables, I_ext, dt=dt, type_indices=type_indices)
        if record_trace:
            trace[t] = state.spikes
    return state, trace


def _column_query(axis: np.ndarray, column: np.ndarray, current: float) -> float:
    """PCHIP along currents that actually have this interval; NaN outside the hull."""
    mask = np.isfinite(column)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    x = axis[mask]
    y = column[mask]
    if current < x[0] or current > x[-1]:
        return float("nan")
    if x.size < 3:
        return float(np.interp(current, x, y))
    from scipy.interpolate import PchipInterpolator

    return float(PchipInterpolator(x, y)(current))


def scheduled_spike_times(table: FittedActivation, current: float, t_end: float = 1000.0) -> np.ndarray:
    """Constant-current L1 spike times (ms) from the fitted schedule.

    Intervals are interpolated across the current axis, then accumulated.
    On a grid node this returns the stored ODE spike times inside the window.
    """
    if table.lut_sched_I is None or table.lut_sched_t is None:
        return np.zeros(0, dtype=np.float64)
    axis = np.asarray(table.lut_sched_I, dtype=np.float64)
    times = np.asarray(table.lut_sched_t, dtype=np.float64)
    I = float(current)
    on_grid = np.flatnonzero(np.isclose(axis, I, rtol=0.0, atol=1e-9))
    if on_grid.size:
        col = times[int(on_grid[0])]
        col = col[np.isfinite(col)]
        return col[col <= t_end + 1e-9]
    t_first = _column_query(axis, times[:, 0], I)
    if not np.isfinite(t_first) or t_first > t_end:
        return np.zeros(0, dtype=np.float64)
    intervals = np.diff(times, axis=1)
    out = [t_first]
    cursor = t_first
    for k in range(intervals.shape[1]):
        step = _column_query(axis, intervals[:, k], I)
        if not np.isfinite(step) or step <= 0.0:
            if k == 0:
                break
            step = _column_query(axis, intervals[:, k - 1], I)
            if not np.isfinite(step) or step <= 0.0:
                break
        cursor = cursor + step
        if cursor > t_end:
            break
        out.append(cursor)
    return np.asarray(out, dtype=np.float64)
