"""C-layer example: register a custom ModelSpec and fit L0 (RFC-002 §3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nrde import ModelSpec
from nrde.fitting.fit import fit_fi, save_fit


@dataclass
class LeakParams:
    """Current-based leaky integrator (mV, ms, nA)."""

    tau: float = 10.0
    R: float = 80.0
    V_rest: float = -70.0
    Vth: float = -50.0
    Vreset: float = -65.0
    t_ref: float = 2.0


def _rhs(t: float, y: np.ndarray, I: float, p: LeakParams) -> np.ndarray:
    del t
    return np.array([(-(y[0] - p.V_rest) + p.R * I) / p.tau], dtype=np.float64)


def _y0(p: LeakParams) -> np.ndarray:
    return np.array([p.V_rest], dtype=np.float64)


def _reset(y: np.ndarray, p: LeakParams) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64).copy()
    out[0] = p.Vreset
    return out


def _vth(p: LeakParams) -> float:
    return p.Vth


def main() -> None:
    spec = ModelSpec(
        name="toy_leak",
        state_dims=1,
        rhs=_rhs,
        y0=_y0,
        Params=LeakParams,
        fitted_targets=["f-i"],
        reset=_reset,
        spike_threshold=_vth,
    )
    spec.register()

    fit = fit_fi(
        "toy_leak",
        type_id="toy_leak",
        I_min=0.1,
        I_max=1.0,
        n_I=12,
        t_total=600.0,
        window=400.0,
        dt=0.05,
        n_validate=4,
    )
    out = Path("artifacts/toy_leak.npz")
    save_fit(fit, out)
    print({"out": str(out), "r2": fit.r2, "layer": fit.layer, "config_hash": fit.config_hash})


if __name__ == "__main__":
    main()
