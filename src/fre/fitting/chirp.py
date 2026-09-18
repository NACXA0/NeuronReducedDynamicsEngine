"""Chirp / impedance |Z(ω)| experiments (FR-3.6, RFC-001)."""

from __future__ import annotations

import numpy as np

from fre.sim import simulate_voltage
from fre.types import ImpedanceReport


def _lockin_gain(sig: np.ndarray, t_ms: np.ndarray, freq_hz: float) -> float:
    omega = 2.0 * np.pi * freq_hz * t_ms / 1000.0
    s = float(np.dot(sig, np.sin(omega)))
    c = float(np.dot(sig, np.cos(omega)))
    return 2.0 * np.hypot(s, c) / max(sig.size, 1)


def chirp_impedance(
    model: str,
    params: dict[str, float] | None = None,
    I0: float = 0.05,
    amp: float = 0.01,
    freqs_hz: np.ndarray | None = None,
    cycles: float = 6.0,
    dt: float = 0.05,
    discard_frac: float = 0.3,
) -> ImpedanceReport:
    """Subthreshold sinusoidal sweep. |Z| = |V_ac| / |I_ac| (mV / nA)."""
    freqs = (
        np.asarray(freqs_hz, dtype=np.float64)
        if freqs_hz is not None
        else np.array([1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    )
    z = np.zeros_like(freqs)
    for i, f in enumerate(freqs.tolist()):
        period_ms = 1000.0 / max(f, 1e-6)
        t_total = period_ms * cycles
        t = np.arange(0.0, t_total, dt)
        I = I0 + amp * np.sin(2.0 * np.pi * f * t / 1000.0)
        V = simulate_voltage(model, params, I, dt=dt, clamp_spikes=True)
        cut = int(discard_frac * t.size)
        vac = V[cut:] - np.mean(V[cut:])
        z[i] = _lockin_gain(vac, t[cut:], f) / max(amp, 1e-12)
    has_peak, peak_hz = detect_resonance(freqs, z)
    notes = f"peak at {peak_hz:.2f} Hz" if has_peak and peak_hz is not None else "no interior |Z| peak"
    return ImpedanceReport(freqs_hz=freqs, z_abs=z, has_peak=has_peak, peak_hz=peak_hz, notes=notes)


def detect_resonance(
    freqs: np.ndarray,
    z: np.ndarray,
    prominence: float = 1.15,
) -> tuple[bool, float | None]:
    if freqs.size < 3:
        return False, None
    k = int(np.argmax(z))
    if k == 0 or k == z.size - 1:
        return False, None
    if z[k] >= prominence * z[0] and z[k] >= prominence * z[-1]:
        return True, float(freqs[k])
    return False, None


def decide_layer(model: str, z: ImpedanceReport | None, has_lut: bool = False) -> tuple[str, str]:
    """FR-3.9: |Z| peak → L2; adapting models → L1; else L0. L3 is v0.2."""
    if z is not None and z.has_peak:
        return "L2", f"impedance peak ({z.notes})"
    if model in {"adexp", "hh"} or has_lut:
        return "L1", f"adaptation/state LUT for {model}"
    return "L0", "steady f-I sufficient"
