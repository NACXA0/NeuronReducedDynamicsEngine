"""CSR SpMV, graph caches, and packed Euler vs the public sim API."""

from __future__ import annotations

import numpy as np

from nrde.engine.rate import run_rate
from nrde.engine.sparse import get_csr, invalidate_graph_cache, sparse_matvec
from nrde.fitting.fit import fit_fi
from nrde.io.connectome import erdos_renyi_graph
from nrde.sim import firing_rate, simulate_spikes
from nrde.types import GraphData


def test_csr_matches_add_at():
    rng = np.random.default_rng(0)
    n, e = 180, 700
    pre = rng.integers(0, n, size=e)
    post = rng.integers(0, n, size=e)
    w = rng.normal(size=e).astype(np.float32)
    x = rng.normal(size=n)
    ref = np.zeros(n, dtype=np.float64)
    np.add.at(ref, post, w.astype(np.float64) * x[pre])
    out = sparse_matvec(np.vstack([pre, post]), w, x, n)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_csr_cache_rebuilds_on_new_weights():
    g = erdos_renyi_graph(20, p=0.2, seed=1)
    a = get_csr(g)
    g.edge_weight = (g.edge_weight * 2.0).astype(np.float32)
    b = get_csr(g)
    assert a is not b
    np.testing.assert_allclose(b.data, get_csr(g).data)
    invalidate_graph_cache(g)
    assert g._csr is None


def test_run_rate_record_trace_off():
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    g = erdos_renyi_graph(12, p=0.1, seed=0)
    state, trace = run_rate(g, [fit, fit], n_steps=8, I_ext=np.full(12, 0.3), record_trace=False)
    assert state.r.shape == (12,)
    assert trace.shape == (0, 12)
    assert np.all(np.isfinite(state.r))


def test_empty_graph_matvec():
    g = GraphData(
        n_nodes=4,
        node_ids=np.arange(4, dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=np.zeros((0,), dtype=np.float32),
        node_type=np.zeros(4, dtype=np.int32),
        type_names=("a",),
    )
    out = sparse_matvec(g.edge_index, g.edge_weight, np.ones(4), 4, csr=get_csr(g))
    np.testing.assert_array_equal(out, np.zeros(4))


def test_packed_lif_matches_analytic_window():
    from nrde.models.lif import LIFParams, analytic_rate

    rate = analytic_rate(0.5, LIFParams())
    num = firing_rate("lif", {}, I=0.5, t_total=400.0, dt=0.05, window=250.0)
    assert abs(num - rate) / rate < 0.15
    spikes = simulate_spikes("lif", {}, I=0.5, t_total=200.0, dt=0.05)
    assert spikes.size >= 1


def test_batch_fi_matches_per_current():
    from nrde.fitting.fit import scan_fi_curve
    from nrde.sim import resolve
    from nrde.sim_kernels import firing_rates_packed, pack_model

    spec, p = resolve("lif", {})
    packed = pack_model(spec.name, p)
    assert packed is not None
    I = np.linspace(0.2, 0.6, 5)
    batch = firing_rates_packed(packed, I, t_total=300.0, dt=0.05, window=180.0)
    one = np.array([firing_rate("lif", {}, float(i), t_total=300.0, dt=0.05, window=180.0) for i in I])
    np.testing.assert_allclose(batch, one, rtol=1e-6, atol=1e-6)
    Ig, rg = scan_fi_curve("lif", I_min=0.2, I_max=0.6, n_I=5, t_total=300.0, window=180.0, dt=0.05)
    np.testing.assert_allclose(rg, batch, rtol=1e-6, atol=1e-6)
    del Ig
