"""Optional PySR symbolic-regression backend (FR-3.7, NFR-11, R8).

PySR is a *refiner*, not the default fitter: maxsize ≤ 20, dim ≤ 2, samples ≤ 1e4,
offline only. The main package must install without Julia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from nrde.models.lif import LIFParams, analytic_rate

PYSR_MAXSIZE = 20
PYSR_MAX_SAMPLES = 10_000
PYSR_MAX_DIM = 2


class FitterBackend(Protocol):
    """Optional refine step after numerical L0/L1/L2 fits (FR-3.7). Offline only."""

    name: str

    def refine_fi(self, I: np.ndarray, r: np.ndarray) -> tuple[str, float]: ...


class PySRUnavailable(RuntimeError):
    """Raised when Julia/PySR is missing or the admission exam retired the backend."""


class PySRFitter:
    name = "pysr"

    def refine_fi(self, I: np.ndarray, r: np.ndarray) -> tuple[str, float]:
        return refine_fi_pysr(I, r)


def pysr_available() -> bool:
    try:
        import pysr  # noqa: F401

        return True
    except ImportError:
        return False


def require_pysr() -> Any:
    """Import pysr or raise with an install hint (NFR-11 / RFC-002 §2.2)."""
    try:
        import pysr

        return pysr
    except ImportError as exc:
        raise ImportError(
            "PySR/Julia is an optional extra. Install with: "
            "pip install 'neuron-reduced-dynamics-engine[pysr]'"
        ) from exc


def _structural_match(expr: str) -> bool:
    text = expr.lower().replace(" ", "")
    return ("log" in text) or ("ln" in text) or ("/" in text)


@dataclass
class AdmissionReport:
    status: str
    expr: str | None
    reason: str
    r2: float | None = None


def admission_exam(
    n_I: int = 64,
    timeout_minutes: float = 30.0,
    niterations: int = 20,
) -> AdmissionReport:
    """LIF closed-form f-I → PySR. Tool calibration, not a design path (RFC-001 D9.5)."""
    del timeout_minutes
    p = LIFParams()
    I_rh = p.gL * (p.Vth - p.EL) + 1e-6
    I = np.linspace(I_rh, I_rh + 0.6, n_I)
    r = np.array([analytic_rate(float(x), p) for x in I])
    if not pysr_available():
        return AdmissionReport(
            status="retired",
            expr=None,
            reason="PySR/Julia not installed; backend retired per NFR-11 / RFC-001 D9.5",
        )
    try:
        expr, r2 = refine_fi_pysr(I, r, niterations=niterations)
    except Exception as exc:  # pragma: no cover - live PySR failures
        return AdmissionReport(status="retired", expr=None, reason=f"PySR exam failed: {exc}")
    ok = _structural_match(expr) and (r2 is not None and r2 >= 0.95)
    return AdmissionReport(
        status="passed" if ok else "retired",
        expr=expr,
        r2=r2,
        reason="recovered log/rational f-I structure" if ok else "expression not structurally consistent",
    )


def refine_fi_pysr(
    I: np.ndarray,
    r: np.ndarray,
    niterations: int = 20,
    maxsize: int = PYSR_MAXSIZE,
) -> tuple[str, float]:
    if I.ndim != 1 or r.ndim != 1:
        raise ValueError(f"PySR refiner is limited to dim ≤ {PYSR_MAX_DIM}; f-I is 1-D")
    if I.size > PYSR_MAX_SAMPLES:
        raise ValueError(f"PySR samples must be ≤ {PYSR_MAX_SAMPLES}")
    if not pysr_available():
        raise PySRUnavailable("install extra: pip install 'neuron-reduced-dynamics-engine[pysr]'")
    from pysr import PySRRegressor  # type: ignore

    X = np.asarray(I, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(r, dtype=np.float64)
    model = PySRRegressor(
        niterations=int(niterations),
        maxsize=int(maxsize),
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp"],
        progress=False,
        verbosity=0,
    )
    model.fit(X, y)
    expr = str(model.sympy())
    pred = np.asarray(model.predict(X), dtype=np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-18 else 1.0 - ss_res / ss_tot
    return expr, r2
