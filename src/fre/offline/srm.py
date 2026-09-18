"""SRM kernel family (L2) with first-order IIR runtime (FR-3.8, NFR-12)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import curve_fit

from fre.engine.sparse import sparse_matvec
from fre.sim import resolve, simulate_voltage
from fre.types import ExpKernel, FittedActivation, GraphData, SpikeState, SRMKernels


def _sum_exp(t: np.ndarray, *theta: float) -> np.ndarray:
    n = len(theta) // 2
    y = np.zeros_like(t, dtype=np.float64)
    for i in range(n):
        a, tau = theta[2 * i], abs(theta[2 * i + 1]) + 1e-6
        y = y + a * np.exp(-t / tau)
    return y


def fit_exponentials(t: np.ndarray, y: np.ndarray, n_exp: int = 2) -> ExpKernel:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p0 = []
    for i in range(n_exp):
        p0.extend([float(y[0]) / n_exp, 10.0 * (i + 1)])
    bounds_lo = [-np.inf] * (2 * n_exp)
    bounds_hi = [np.inf] * (2 * n_exp)
    for i in range(n_exp):
        bounds_lo[2 * i + 1] = 0.2
        bounds_hi[2 * i + 1] = 500.0
    try:
        popt, _ = curve_fit(_sum_exp, t, y, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=8000)
    except (RuntimeError, ValueError):
        popt = np.asarray(p0, dtype=np.float64)
    amps = np.array(popt[0::2], dtype=np.float64)
    taus = np.abs(np.array(popt[1::2], dtype=np.float64)) + 1e-6
    return ExpKernel(amps=amps, taus=taus)


def fit_srm_kernels(
    model: str = "adexp",
    params: dict[str, float] | None = None,
    dt: float = 0.05,
    t_kernel: float = 80.0,
    pulse: float = 0.2,
) -> SRMKernels:
    """Impulse responses of the named ODE → κ, η, θ as exponential sums."""
    spec, p = resolve(model, params)
    t = np.arange(0.0, t_kernel, dt)
    I_imp = np.zeros_like(t)
    I_imp[0] = pulse / max(dt, 1e-6)
    V = simulate_voltage(model, params, I_imp, dt=dt, clamp_spikes=True)
    V0 = float(spec.y0(p)[0])
    kappa = fit_exponentials(t, V - V0, n_exp=2)

    y = spec.reset(spec.y0(p), p) if spec.hybrid_reset() else spec.y0(p)
    rec = np.empty_like(t)
    rec[0] = float(y[0])
    for k in range(1, t.size):
        y = y + dt * spec.rhs(t[k], y, 0.0, p)
        rec[k] = float(y[0])
    eta = fit_exponentials(t, rec - rec[-1], n_exp=2)
    theta = ExpKernel(
        amps=np.array([5.0, 1.0], dtype=np.float64),
        taus=np.array([float(getattr(p, "t_ref", 2.0)) + 2.0, 20.0], dtype=np.float64),
    )
    v_rh = float(getattr(p, "Vth", getattr(p, "VT", -50.0)))
    return SRMKernels(kappa=kappa, eta=eta, theta=theta, v_rh=v_rh)


def attach_srm(fit: FittedActivation, kernels: SRMKernels) -> FittedActivation:
    fit.srm = kernels
    if fit.layer in {"L0", "L1"}:
        fit.layer = "L2"
    return fit


def init_srm_state(n_nodes: int, kernels: SRMKernels, V0: float = -70.0) -> SpikeState:
    nk = int(kernels.kappa.amps.size)
    ne = int(kernels.eta.amps.size)
    nt = int(kernels.theta.amps.size)
    return SpikeState(
        V=np.full(n_nodes, V0, dtype=np.float64),
        t_since=np.full(n_nodes, 10.0, dtype=np.float64),
        spikes=np.zeros(n_nodes, dtype=bool),
        I_syn=np.zeros(n_nodes, dtype=np.float64),
        s_kappa=np.zeros((n_nodes, nk), dtype=np.float64),
        s_eta=np.zeros((n_nodes, ne), dtype=np.float64),
        s_theta=np.zeros((n_nodes, nt), dtype=np.float64),
    )


def step_srm(
    state: SpikeState,
    graph: GraphData,
    kernels: SRMKernels,
    I_ext: np.ndarray,
    dt: float = 1.0,
    tau_syn: float = 2.0,
) -> SpikeState:
    """One SRM step: O(1) per kernel per neuron (NFR-12)."""
    if state.s_kappa is None or state.s_eta is None or state.s_theta is None:
        state = init_srm_state(graph.n_nodes, kernels)
    decay = float(np.exp(-dt / tau_syn))
    pulse = sparse_matvec(
        graph.edge_index,
        graph.edge_weight,
        state.spikes.astype(np.float64),
        graph.n_nodes,
    )
    I_syn = state.I_syn * decay + pulse
    I_total = I_syn + np.asarray(I_ext, dtype=np.float64)
    dk = kernels.kappa.decay(dt)
    de = kernels.eta.decay(dt)
    dth = kernels.theta.decay(dt)
    s_k = state.s_kappa * dk + np.outer(I_total, kernels.kappa.amps)
    s_e = state.s_eta * de
    s_th = state.s_theta * dth
    u = s_k.sum(axis=1) - s_e.sum(axis=1)
    thresh = kernels.v_rh + s_th.sum(axis=1)
    spikes = u >= thresh
    if np.any(spikes):
        s_e[spikes] = s_e[spikes] + kernels.eta.amps
        s_th[spikes] = s_th[spikes] + kernels.theta.amps
    t_since = np.where(spikes, 0.0, state.t_since + dt)
    return replace(
        state,
        V=u,
        spikes=spikes,
        I_syn=I_syn,
        t_since=t_since,
        s_kappa=s_k,
        s_eta=s_e,
        s_theta=s_th,
    )
