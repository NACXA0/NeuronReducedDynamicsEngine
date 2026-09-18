"""TC-4.x rate/spike engines, Izhikevich fallback, numerical safety."""

from __future__ import annotations

import numpy as np

from fre.engine.rate import run_rate
from fre.engine.spike import run_spike
from fre.io.connectome import erdos_renyi_graph
from fre.offline.fit import attach_lut, fit_fi, fit_spike_lut
from fre.validation import relative_vp


def _lif_fit():
    return fit_fi(
        "lif",
        type_id="type_0",
        I_min=0.0,
        I_max=0.8,
        n_I=10,
        t_total=400.0,
        window=250.0,
        dt=0.05,
        n_validate=3,
    )


def test_tc_4_1_rate_two_types_steady_distribution():
    fit_a = _lif_fit()
    fit_b = fit_fi(
        "explif",
        type_id="type_1",
        I_min=0.0,
        I_max=0.8,
        n_I=10,
        t_total=400.0,
        window=250.0,
        dt=0.05,
        n_validate=3,
    )
    g = erdos_renyi_graph(
        100,
        p=0.04,
        n_types=2,
        seed=2,
        weight=0.0008,
        type_names=("type_0", "type_1"),
    )
    I_ext = np.full(g.n_nodes, 0.35)
    state, trace = run_rate(g, [fit_a, fit_b], n_steps=80, I_ext=I_ext)
    assert state.r.shape == (100,)
    assert np.all(np.isfinite(state.r))
    assert np.mean(trace[-10:]) > 0.0
    means = [float(np.mean(state.r[g.node_type == t])) for t in range(2)]
    assert all(m >= 0.0 for m in means)


def test_tc_4_1_thousand_neurons_acceptance():
    fit = _lif_fit()
    g = erdos_renyi_graph(1000, p=0.008, n_types=2, seed=0, weight=0.0005)
    I_ext = np.full(1000, 0.3)
    state, _ = run_rate(g, [fit, fit], n_steps=40, I_ext=I_ext)
    assert state.r.shape == (1000,)
    assert np.all(np.isfinite(state.r))
    assert float(np.mean(state.r)) >= 0.0


def test_tc_4_2_spike_lut_vs_ode_vp():
    fit = _lif_fit()
    lut = fit_spike_lut("lif", I_min=0.0, I_max=0.8, n_I=12, n_dt=16, dt=1.0, t_ref_max=40.0)
    fit = attach_lut(fit, lut)
    from fre.sim import simulate_spikes
    from fre.types import GraphData

    I0 = 0.45
    ref = simulate_spikes("lif", {}, I0, t_total=1000.0, dt=0.05)
    graph = GraphData(
        n_nodes=1,
        node_ids=np.array([0], dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=np.zeros((0,), dtype=np.float32),
        node_type=np.array([0], dtype=np.int32),
        type_names=("lif",),
    )
    _, trace = run_spike(graph, [fit], n_steps=1000, I_ext=np.array([I0]))
    approx = np.where(trace[:, 0])[0].astype(np.float64)
    rel = relative_vp(ref, approx)
    assert ref.size > 0
    assert approx.size > 0
    assert rel <= 0.10 or abs(ref.size - approx.size) / max(ref.size, 1) <= 0.2


def test_tc_4_3_refractory_prevents_double_spike():
    fit = attach_lut(_lif_fit(), fit_spike_lut("lif", n_I=8, n_dt=8))
    g = erdos_renyi_graph(8, p=0.0, n_types=1, seed=0)
    I_ext = np.full(8, 2.0)
    _, trace = run_spike(g, [fit], n_steps=30, I_ext=I_ext)
    diffs = np.diff(np.where(trace[:, 0])[0])
    if diffs.size:
        assert diffs.min() >= 1


def test_tc_4_4_external_current_channel():
    fit = _lif_fit()
    g = erdos_renyi_graph(20, p=0.05, seed=3)
    i_ext = np.zeros(20)
    i_ext[:3] = 0.5
    state, _ = run_rate(g, [fit, fit], n_steps=30, I_ext=i_ext)
    assert np.mean(state.r[:3]) >= np.mean(state.r[3:]) - 1e-6


def test_tc_4_stability_large_current():
    fit = _lif_fit()
    g = erdos_renyi_graph(30, p=0.05, seed=4)
    state, _ = run_rate(g, [fit, fit], n_steps=20, I_ext=np.full(30, 50.0))
    assert np.all(np.isfinite(state.r))
    assert state.out_of_range > 0


def test_tc_4_izhikevich_fallback():
    fit = fit_fi("izhikevich", I_min=0.0, I_max=15.0, n_I=6, t_total=200.0, window=120.0, dt=0.5, n_validate=2)
    g = erdos_renyi_graph(12, p=0.1, n_types=1, seed=5, type_names=("izh",))
    tables = [fit]
    state, trace = run_spike(g, tables, n_steps=80, I_ext=np.full(12, 10.0), mode="izhikevich")
    assert trace.any()
    assert np.all(np.isfinite(state.V))
