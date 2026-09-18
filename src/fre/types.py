"""Public types shared across FRE modules (DD-1, DD-2, DD-3 / RFC-001)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

NPZ_SCHEMA_VERSION = 2
NPZ_SCHEMA_COMPAT = 1

# Shiu et al. 2024: single free parameter, voltage jump per synapse.
SHIU_W_SYN_MV = 0.275
# Current-based equivalent: I = ΔV * C / dt with C=0.1 nF, dt=1 ms → 0.0275 nA / synapse.
SHIU_ALPHA_NA = SHIU_W_SYN_MV * 0.1 / 1.0

TIER_DEFAULT = "default"
TIER_LITERATURE = "literature"
TIER_RESERVED = "reserved"


@dataclass
class TypeEntry:
    type_id: str
    model: str
    params: dict[str, float] = field(default_factory=dict)
    ref: str | None = None
    quality: str = "unknown"
    tier: str = TIER_DEFAULT
    reduction_benefit: str = "control"


@dataclass
class ExpKernel:
    """Sum of exponentials Σ a_i exp(-t/τ_i). IIR: s <- exp(-dt/τ) s + a x."""

    amps: np.ndarray
    taus: np.ndarray

    def decay(self, dt: float) -> np.ndarray:
        taus = np.maximum(np.asarray(self.taus, dtype=np.float64), 1e-6)
        return np.exp(-float(dt) / taus)


@dataclass
class SRMKernels:
    """L2 SRM kernels κ (input), η (after-spike), θ (threshold). FR-3.8."""

    kappa: ExpKernel
    eta: ExpKernel
    theta: ExpKernel
    v_rh: float = -50.0


@dataclass
class ImpedanceReport:
    freqs_hz: np.ndarray
    z_abs: np.ndarray
    has_peak: bool
    peak_hz: float | None
    notes: str = ""


@dataclass
class FittedActivation:
    """Type-bound reduced F: L0 f-I, optional L1 LUT, L2 SRM, optional PySR expr."""

    schema_version: int
    type_id: str
    model: str
    params: dict[str, float]
    method: str
    I_grid: np.ndarray
    r_grid: np.ndarray
    I_onset: float
    r2: float
    mse: float
    quality: str
    notes: str = ""
    layer: str = "L0"
    level_decision: str = ""
    lut_I: np.ndarray | None = None
    lut_dt: np.ndarray | None = None
    lut_spike: np.ndarray | None = None
    lut_V: np.ndarray | None = None
    lut_hit_rate: float | None = None
    srm: SRMKernels | None = None
    z_freqs: np.ndarray | None = None
    z_abs: np.ndarray | None = None
    pysr_expr: str | None = None
    fallback_ode: bool = False
    # Provenance (optional; filled on save). M2+ prefers type-keyed filenames.
    type_ids: tuple[str, ...] = ()
    git_commit: str | None = None
    config_hash: str | None = None

    def eval_rate(self, I: np.ndarray) -> tuple[np.ndarray, int]:
        I = np.asarray(I, dtype=np.float64)
        oor = int(np.count_nonzero((I < self.I_grid[0]) | (I > self.I_grid[-1])))
        I_clip = np.clip(I, self.I_grid[0], self.I_grid[-1])
        r = np.interp(I_clip, self.I_grid, self.r_grid)
        r = np.where(I_clip < self.I_onset, 0.0, r)
        r = np.clip(r, 0.0, float(np.max(self.r_grid)) if self.r_grid.size else 0.0)
        return r.astype(np.float64), oor


@dataclass
class GraphData:
    n_nodes: int
    node_ids: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    node_type: np.ndarray
    type_names: tuple[str, ...]
    nt_type: np.ndarray | None = None
    region: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    fallback_mask: np.ndarray | None = None

    def type_name_of(self, index: int) -> str:
        return self.type_names[int(self.node_type[index])]


@dataclass
class RateState:
    r: np.ndarray
    out_of_range: int = 0
    y_ode: np.ndarray | None = None


@dataclass
class SpikeState:
    V: np.ndarray
    t_since: np.ndarray
    spikes: np.ndarray
    I_syn: np.ndarray
    u: np.ndarray | None = None
    s_kappa: np.ndarray | None = None
    s_eta: np.ndarray | None = None
    s_theta: np.ndarray | None = None
    out_of_range: int = 0


def as_float_params(params: Mapping[str, Any] | None) -> dict[str, float]:
    if not params:
        return {}
    return {str(k): float(v) for k, v in params.items()}
