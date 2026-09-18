"""Optional JAX f-I and PyTorch CSR engine backends."""

from __future__ import annotations

import numpy as np
import pytest

from nrde.engine.backends import resolve_engine_backend, torch_available, torch_cuda_available
from nrde.engine.rate import run_rate
from nrde.fitting.fit import scan_fi_curve
from nrde.io.connectome import erdos_renyi_graph
from nrde.sim import firing_rate, resolve
from nrde.sim_jax import jax_available
from nrde.sim_kernels import firing_rates_packed, pack_model


def test_resolve_engine_backend_numpy_default(monkeypatch):
    monkeypatch.delenv("NRDE_ENGINE", raising=False)
    assert resolve_engine_backend(None) == "numpy"
    assert resolve_engine_backend("numpy") == "numpy"
    with pytest.raises(ValueError, match="Unknown"):
        resolve_engine_backend("cuda-kernel")


def test_jax_backend_requires_extra():
    if jax_available():
        pytest.skip("jax installed")
    with pytest.raises(ImportError, match="jax"):
        scan_fi_curve("lif", n_I=3, t_total=80.0, window=50.0, dt=0.1, backend="jax")


def test_torch_run_rate_requires_extra():
    if torch_available():
        pytest.skip("torch installed")
    from nrde.types import NPZ_SCHEMA_VERSION, FittedActivation

    fit = FittedActivation(
        schema_version=NPZ_SCHEMA_VERSION,
        type_id="x",
        model="lif",
        params={},
        method="linear",
        I_grid=np.array([0.0, 1.0]),
        r_grid=np.array([0.0, 50.0]),
        I_onset=0.1,
        r2=1.0,
        mse=0.0,
        quality="ok",
    )
    g = erdos_renyi_graph(6, p=0.2, seed=0, n_types=1)
    with pytest.raises(ImportError, match="torch"):
        run_rate(g, [fit], n_steps=1, I_ext=np.zeros(6), backend="torch")


@pytest.mark.jax
def test_jax_fi_matches_packed():
    if not jax_available():
        pytest.skip("jax extra not installed")
    spec, p = resolve("lif", {})
    packed = pack_model(spec.name, p)
    assert packed is not None
    I = np.linspace(0.25, 0.55, 4)
    ref = firing_rates_packed(packed, I, t_total=250.0, dt=0.05, window=150.0)
    _, jax_r = scan_fi_curve(
        "lif",
        I_min=0.25,
        I_max=0.55,
        n_I=4,
        t_total=250.0,
        window=150.0,
        dt=0.05,
        backend="jax",
    )
    np.testing.assert_allclose(jax_r, ref, rtol=1e-5, atol=1e-5)
    one = firing_rate("lif", {}, float(I[1]), t_total=250.0, dt=0.05, window=150.0)
    assert abs(float(jax_r[1]) - one) <= max(1.0, 0.05 * one)


@pytest.mark.torch
def test_torch_cpu_rate_parity():
    if not torch_available():
        pytest.skip("torch extra not installed")
    from nrde.fitting.fit import fit_fi

    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    g = erdos_renyi_graph(24, p=0.12, seed=4, n_types=2)
    I_ext = np.full(g.n_nodes, 0.3)
    np_state, _ = run_rate(g, [fit, fit], n_steps=12, I_ext=I_ext, record_trace=False, backend="numpy")
    from nrde.engine.torch_rate import run_rate_torch

    th_state, _ = run_rate_torch(
        g, [fit, fit], n_steps=12, I_ext=I_ext, record_trace=False, device="cpu"
    )
    np.testing.assert_allclose(th_state.r, np_state.r, rtol=2e-4, atol=2e-4)


@pytest.mark.torch
def test_nfr1_torch_cuda_report():
    if not torch_cuda_available():
        pytest.skip("CUDA not available")
    import time

    from nrde.fitting.fit import fit_fi
    from nrde.types import GraphData

    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.6)
    rng = np.random.default_rng(0)
    n = 10_000
    n_edges = 40_000
    g = GraphData(
        n_nodes=n,
        node_ids=np.arange(n, dtype=np.int64),
        edge_index=np.vstack([rng.integers(0, n, n_edges), rng.integers(0, n, n_edges)]),
        edge_weight=np.full(n_edges, 0.0002, dtype=np.float32),
        node_type=rng.integers(0, 2, size=n, dtype=np.int32),
        type_names=("a", "b"),
    )
    t0 = time.perf_counter()
    state, _ = run_rate(
        g, [fit, fit], n_steps=20, I_ext=np.full(n, 0.25), record_trace=False, backend="torch"
    )
    elapsed = time.perf_counter() - t0
    assert np.all(np.isfinite(state.r))
    realtime = 0.020 / max(elapsed, 1e-9)
    print(f"torch CUDA 10k x 20 steps: {elapsed:.3f}s wall, {realtime:.2f}x realtime")
