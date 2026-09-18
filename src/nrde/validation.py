"""Single-cell, circuit, and FlyWire-aligned validation (FR-6, NFR-4/5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nrde.engine.rate import run_rate
from nrde.engine.spike import run_spike, scheduled_spike_times
from nrde.sim import firing_rate, simulate_spikes
from nrde.types import FittedActivation, GraphData

VP_Q_PER_MS = 0.1  # q = 1/10 ms = 0.1 / ms
VP_WINDOW_MS = 1000.0
REL_VP_GATE = 0.10


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-18:
        return 1.0 if ss_res <= 1e-18 else 0.0
    return 1.0 - ss_res / ss_tot


def victor_purpura(t1: np.ndarray, t2: np.ndarray, q: float = VP_Q_PER_MS) -> float:
    a = np.sort(np.asarray(t1, dtype=np.float64))
    b = np.sort(np.asarray(t2, dtype=np.float64))
    n, m = a.size, b.size
    D = np.zeros((n + 1, m + 1), dtype=np.float64)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = q * abs(a[i - 1] - b[j - 1])
            D[i, j] = min(D[i - 1, j] + 1.0, D[i, j - 1] + 1.0, D[i - 1, j - 1] + cost)
    return float(D[n, m])


def relative_vp(t1: np.ndarray, t2: np.ndarray, q: float = VP_Q_PER_MS) -> float:
    d = victor_purpura(t1, t2, q=q)
    denom = max(t1.size + t2.size, 1)
    return float(d / denom)


@dataclass
class SingleCellReport:
    mse: float
    r2: float
    relative_vp: float | None
    nfr4_pass: bool
    nfr5_pass: bool | None
    notes: str = ""


def validate_single(
    fit: FittedActivation,
    I_test: np.ndarray | None = None,
    t_total: float = 1200.0,
    window: float = 800.0,
    dt: float = 0.05,
) -> SingleCellReport:
    if I_test is None:
        I_test = np.linspace(fit.I_grid[0], fit.I_grid[-1], 6)
    r_true = np.array(
        [
            firing_rate(fit.model, fit.params, float(I), t_total=t_total, dt=dt, window=window)
            for I in I_test
        ]
    )
    r_pred, _ = fit.eval_rate(I_test)
    mse = float(np.mean((r_true - r_pred) ** 2))
    r2 = r2_score(r_true, r_pred)
    rel_vp = None
    nfr5 = None
    if fit.lut_I is not None:
        I0 = float(I_test[min(2, I_test.size - 1)])
        ref = simulate_spikes(fit.model, fit.params, I0, t_total=VP_WINDOW_MS, dt=dt)
        if fit.lut_sched_t is not None:
            approx = scheduled_spike_times(fit, I0, t_end=VP_WINDOW_MS)
        else:
            graph = GraphData(
                n_nodes=1,
                node_ids=np.array([0], dtype=np.int64),
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_weight=np.zeros((0,), dtype=np.float32),
                node_type=np.array([0], dtype=np.int32),
                type_names=(fit.type_id,),
            )
            _, trace = run_spike(graph, [fit], n_steps=int(VP_WINDOW_MS), I_ext=np.array([I0]))
            approx = np.where(trace[:, 0])[0].astype(np.float64)
        rel_vp = relative_vp(ref, approx)
        nfr5 = bool(rel_vp <= REL_VP_GATE)
    return SingleCellReport(
        mse=mse,
        r2=r2,
        relative_vp=rel_vp,
        nfr4_pass=bool(r2 >= 0.98),
        nfr5_pass=nfr5,
    )


def psth(spike_trace: np.ndarray, bin_ms: int = 20) -> np.ndarray:
    n_steps, n_cells = spike_trace.shape
    n_bins = max(1, n_steps // bin_ms)
    usable = n_bins * bin_ms
    data = spike_trace[:usable].reshape(n_bins, bin_ms, n_cells)
    return data.sum(axis=1) / (bin_ms / 1000.0)


def validate_circuit(
    graph: GraphData,
    tables: list[FittedActivation],
    I_ext: np.ndarray,
    n_steps: int = 200,
) -> dict[str, float]:
    """Compare rate engine vs a uniform-LIF ablation on the same graph."""
    state, trace = run_rate(graph, tables, n_steps=n_steps, I_ext=I_ext)
    mean_r = float(np.mean(state.r))
    std_r = float(np.std(state.r))
    return {
        "mean_rate": mean_r,
        "std_rate": std_r,
        "out_of_range": float(state.out_of_range),
        "trace_mean": float(np.mean(trace[-20:])),
    }


def uniform_lif_ablation(
    graph: GraphData,
    typed_tables: list[FittedActivation],
    I_ext: np.ndarray,
    n_steps: int = 200,
) -> dict[str, float]:
    """Same subgraph, every cell uses the first table (uniform LIF-style F)."""
    uniform = [typed_tables[0] for _ in typed_tables]
    typed = validate_circuit(graph, typed_tables, I_ext, n_steps=n_steps)
    uni = validate_circuit(graph, uniform, I_ext, n_steps=n_steps)
    return {
        "typed_mean_rate": typed["mean_rate"],
        "uniform_mean_rate": uni["mean_rate"],
        "mean_rate_delta": abs(typed["mean_rate"] - uni["mean_rate"]),
        "distinguishable": abs(typed["mean_rate"] - uni["mean_rate"]) > 0.5,
    }


def validate_hopkins_flywire(note: str | None = None) -> dict[str, str]:
    """System-level comparison is FlyWire-only (FR-6.3). MaleCNS is load/perf."""
    return {
        "dataset": "flywire",
        "status": "stub",
        "note": note
        or (
            "Shiu et al. Nature 2024 uses the FlyWire adult brain. "
            "Do not mix MaleCNS body IDs. Provide a FlyWire subgraph to run a qualitative PSTH check."
        ),
    }
