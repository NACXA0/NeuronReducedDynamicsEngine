"""Network-level alpha calibration (FR-6, DD-6, R1). Bound to M2."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from nrde.engine.rate import run_rate
from nrde.types import SHIU_ALPHA_NA, SHIU_W_SYN_MV, FittedActivation, GraphData


def shiu_alpha_na(c_nF: float = 0.1, dt_ms: float = 1.0) -> float:
    """Map Shiu W_syn (mV / synapse) to current-based α (nA / synapse)."""
    return float(SHIU_W_SYN_MV) * float(c_nF) / float(dt_ms)


def apply_shiu_weights(graph: GraphData, synapse_counts: np.ndarray | None = None) -> GraphData:
    """Reset |w| so |w| = α_Shiu × synapse_count, preserving sign."""
    if synapse_counts is None:
        signs = np.sign(graph.edge_weight)
        signs[signs == 0] = 1.0
        counts = np.maximum(np.abs(graph.edge_weight) / max(SHIU_ALPHA_NA, 1e-12), 1.0)
    else:
        signs = np.sign(graph.edge_weight)
        signs[signs == 0] = 1.0
        counts = np.asarray(synapse_counts, dtype=np.float64)
    graph.edge_weight = (signs * counts * SHIU_ALPHA_NA).astype(np.float32)
    graph.meta = dict(graph.meta)
    graph.meta["w_syn_mv"] = SHIU_W_SYN_MV
    graph.meta["alpha_na"] = SHIU_ALPHA_NA
    return graph


def calibrate_alpha(
    graph: GraphData,
    tables: list[FittedActivation],
    target_rates: np.ndarray,
    I_ext: np.ndarray | None = None,
    n_steps: int = 50,
    alpha_bounds: tuple[float, float] = (0.05, 20.0),
    start_from_shiu: bool = True,
) -> tuple[float, float]:
    """Find global scale s on top of current weights (Shiu start recommended).

    Acceptance (RFC-001 §5.4): after calibration, mean |rate - target| / target ≤ 10%
    on the labelled circuit.
    """
    if start_from_shiu and graph.meta.get("w_syn_mv") != SHIU_W_SYN_MV:
        apply_shiu_weights(graph)
    target = np.asarray(target_rates, dtype=np.float64)
    I_ext = np.zeros(graph.n_nodes) if I_ext is None else np.asarray(I_ext, dtype=np.float64)
    base_w = graph.edge_weight.copy()

    def mse(scale: float) -> float:
        graph.edge_weight = (base_w * float(scale)).astype(np.float32)
        state, _ = run_rate(graph, tables, n_steps=n_steps, I_ext=I_ext, record_trace=False)
        return float(np.mean((state.r - target) ** 2))

    result = minimize_scalar(mse, bounds=alpha_bounds, method="bounded", options={"xatol": 1e-3})
    graph.edge_weight = base_w
    return float(result.x), float(result.fun)


def relative_rate_error(rates: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    denom = np.maximum(np.abs(target), 1.0)
    return float(np.mean(np.abs(np.asarray(rates) - target) / denom))


def apply_alpha_scale(graph: GraphData, scale: float) -> GraphData:
    graph.edge_weight = (graph.edge_weight * float(scale)).astype(np.float32)
    graph.meta = dict(graph.meta)
    graph.meta["alpha_scale"] = float(scale)
    return graph
