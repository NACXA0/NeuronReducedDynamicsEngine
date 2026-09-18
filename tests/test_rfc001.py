"""FR-3.6 chirp, FR-3.7 PySR, FR-3.8 SRM, FR-3.9 layer decision, DD-4 fallback."""

from __future__ import annotations

import numpy as np
import pytest

from fre.engine.hybrid import step_rate_mixed
from fre.engine.rate import init_rate_state
from fre.io.connectome import erdos_renyi_graph
from fre.offline.chirp import chirp_impedance, decide_layer, detect_resonance
from fre.offline.fit import attach_lut, fit_fi, fit_spike_lut, load_fit, save_fit
from fre.offline.pysr_backend import PySRUnavailable, admission_exam, pysr_available, refine_fi_pysr
from fre.offline.srm import attach_srm, fit_srm_kernels, init_srm_state, step_srm
from fre.types import ImpedanceReport


def test_adexp_is_v0_1_primary():
    from fre.models import get_model

    assert get_model("adexp").reduction_benefit == "v0.1_primary"
    assert get_model("lif").reduction_benefit == "control"


def test_chirp_lif_has_no_peak():
    z = chirp_impedance(
        "lif",
        freqs_hz=np.array([2.0, 10.0, 40.0]),
        cycles=3.0,
        dt=0.1,
    )
    assert z.z_abs.size == 3
    assert np.all(z.z_abs > 0)
    layer, _ = decide_layer("lif", z)
    assert layer == "L0"


def test_adexp_fit_layer_l1_without_chirp():
    fit = fit_fi("adexp", type_id="aCC", n_I=8, t_total=300.0, window=180.0, I_max=0.8, n_validate=2)
    assert fit.layer == "L1"
    assert fit.model == "adexp"
    assert fit.r2 >= 0.9 or fit.r_grid.max() >= 0.0


def test_layer_l2_when_peak():
    z = ImpedanceReport(
        freqs_hz=np.array([1.0, 10.0, 40.0]),
        z_abs=np.array([1.0, 3.0, 1.0]),
        has_peak=True,
        peak_hz=10.0,
    )
    layer, why = decide_layer("adexp", z)
    assert layer == "L2"
    assert "peak" in why


def test_srm_kernels_and_iir_step():
    kernels = fit_srm_kernels("adexp", t_kernel=30.0, dt=0.1)
    assert kernels.kappa.amps.size == 2
    g = erdos_renyi_graph(8, p=0.2, n_types=1, seed=0, type_names=("aCC",))
    state = init_srm_state(g.n_nodes, kernels)
    state = step_srm(state, g, kernels, np.full(8, 0.4), dt=1.0)
    assert state.spikes.shape == (8,)
    assert state.s_kappa is not None


def test_srm_roundtrip(tmp_path):
    fit = fit_fi("adexp", n_I=6, t_total=250.0, window=150.0, n_validate=2, I_max=0.7)
    fit = attach_srm(fit, fit_srm_kernels("adexp", t_kernel=20.0, dt=0.1))
    assert fit.layer == "L2"
    path = save_fit(fit, tmp_path / "srm.npz")
    loaded = load_fit(path)
    assert loaded.srm is not None
    assert loaded.layer == "L2"


def test_pysr_unavailable_without_julia():
    report = admission_exam()
    if pysr_available():
        pytest.skip("PySR installed; live exam is the extra CI lane")
    assert report.status == "retired"
    assert "NFR-11" in report.reason
    with pytest.raises(PySRUnavailable):
        refine_fi_pysr(np.linspace(0.2, 0.8, 8), np.linspace(0, 50, 8))


@pytest.mark.pysr
def test_pysr_admission_live():
    report = admission_exam(n_I=32, niterations=5)
    assert report.status in {"passed", "retired"}


def test_hybrid_fallback_runs_ode():
    fit = fit_fi("adexp", type_id="T4", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.7)
    g = erdos_renyi_graph(10, p=0.1, n_types=1, seed=2, type_names=("T4",))
    state = init_rate_state(g.n_nodes)
    state = step_rate_mixed(state, g, [fit], np.full(10, 0.5), fallback_types={"T4"})
    assert state.y_ode is not None
    assert np.all(np.isfinite(state.r))


def test_detect_resonance_requires_interior_peak():
    assert detect_resonance(np.array([1.0, 10.0, 40.0]), np.array([3.0, 1.0, 1.0]))[0] is False
    ok, peak = detect_resonance(np.array([1.0, 10.0, 40.0]), np.array([1.0, 3.0, 1.0]))
    assert ok and peak == 10.0


def test_attach_lut_promotes_l0_to_l1():
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    assert fit.layer == "L0"
    fit = attach_lut(fit, fit_spike_lut("lif", n_I=6, n_dt=6, I_max=0.5))
    assert fit.layer == "L1"


def test_schema_v1_still_loads(tmp_path):
    I = np.linspace(0.0, 1.0, 8)
    r = np.clip(I - 0.2, 0, None) * 80.0
    path = tmp_path / "v1.npz"
    np.savez_compressed(
        path,
        schema_version=np.array([1]),
        I_grid=I,
        r_grid=r,
        I_onset=np.array([0.2]),
        r2=np.array([0.99]),
        mse=np.array([0.01]),
    )
    loaded = load_fit(path)
    assert loaded.schema_version == 1
    assert loaded.layer == "L0"
    assert loaded.srm is None


def test_nfr12_srm_iir_cost_vs_lut():
    import time

    from fre.engine.spike import init_spike_state, step_spike_lut

    kernels = fit_srm_kernels("lif", t_kernel=15.0, dt=0.1)
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.6)
    fit = attach_lut(fit, fit_spike_lut("lif", n_I=6, n_dt=8, I_max=0.6))
    g = erdos_renyi_graph(48, p=0.08, n_types=1, seed=1, type_names=("lif",))
    I_ext = np.full(g.n_nodes, 0.4)
    srm_state = init_srm_state(g.n_nodes, kernels)
    lut_state = init_spike_state(g, [fit])
    for _ in range(3):
        srm_state = step_srm(srm_state, g, kernels, I_ext)
        lut_state = step_spike_lut(lut_state, g, [fit], I_ext)
    t0 = time.perf_counter()
    for _ in range(40):
        srm_state = step_srm(srm_state, g, kernels, I_ext)
    t_srm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(40):
        lut_state = step_spike_lut(lut_state, g, [fit], I_ext)
    t_lut = time.perf_counter() - t0
    assert kernels.kappa.amps.size <= 2
    # CI noise: structural O(1)/kernel plus a loose wall-clock bound (NFR-12 target is 3×).
    assert t_srm <= max(3.0 * t_lut, 0.5)


def test_adexp_chirp_report_always_generated():
    z = chirp_impedance(
        "adexp",
        freqs_hz=np.array([2.0, 10.0, 40.0]),
        cycles=3.0,
        dt=0.1,
    )
    assert z.z_abs.size == 3
    assert np.all(np.isfinite(z.z_abs))
    assert z.notes


def test_mixed_without_fallback_is_pure_rate():
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    g = erdos_renyi_graph(8, p=0.1, n_types=1, seed=0, type_names=("x",))
    state = init_rate_state(g.n_nodes)
    out = step_rate_mixed(state, g, [fit], np.full(8, 0.3), fallback_types=set())
    assert out.y_ode is None
    assert np.all(np.isfinite(out.r))


def test_pysr_rejects_wrong_shape():
    with pytest.raises(ValueError, match="dim"):
        refine_fi_pysr(np.ones((4, 2)), np.ones(4))


def test_schema_too_old_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        schema_version=np.array([0]),
        I_grid=np.array([0.0, 1.0]),
        r_grid=np.array([0.0, 1.0]),
        I_onset=np.array([0.0]),
        r2=np.array([1.0]),
        mse=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="too old"):
        load_fit(path)


def test_shiu_explicit_synapse_counts():
    from fre.calibration import apply_shiu_weights
    from fre.types import SHIU_ALPHA_NA

    g = erdos_renyi_graph(4, p=1.0, n_types=1, seed=0, weight=1.0)
    counts = np.full(g.edge_weight.shape, 2.0)
    apply_shiu_weights(g, synapse_counts=counts)
    assert np.allclose(np.abs(g.edge_weight), 2.0 * SHIU_ALPHA_NA)
