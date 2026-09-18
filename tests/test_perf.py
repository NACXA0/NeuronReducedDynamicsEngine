"""NFR-1/3 report-style tests: 10k rate step and MaleCNS-like Feather load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from nrde.engine.rate import run_rate
from nrde.fitting.fit import fit_fi
from nrde.io.connectome import load_connectome


@pytest.mark.slow
def test_nfr1_10k_rate_runs():
    fit = fit_fi("lif", n_I=6, t_total=250.0, window=150.0, n_validate=2, I_max=0.6)
    rng = np.random.default_rng(0)
    n = 10_000
    n_edges = 40_000
    pre = rng.integers(0, n, size=n_edges)
    post = rng.integers(0, n, size=n_edges)
    from nrde.types import GraphData

    g = GraphData(
        n_nodes=n,
        node_ids=np.arange(n, dtype=np.int64),
        edge_index=np.vstack([pre, post]),
        edge_weight=np.full(n_edges, 0.0002, dtype=np.float32),
        node_type=rng.integers(0, 2, size=n, dtype=np.int32),
        type_names=("a", "b"),
    )
    import time

    t0 = time.perf_counter()
    state, _ = run_rate(g, [fit, fit], n_steps=20, I_ext=np.full(n, 0.25))
    elapsed = time.perf_counter() - t0
    assert np.all(np.isfinite(state.r))
    sim_ms = 20.0
    print(f"10k x 20 steps: {elapsed:.3f}s wall, {sim_ms:.0f} ms sim")


def test_nfr3_malecns_like_feather_load_and_step(tmp_path: Path):
    n = 64
    rng = np.random.default_rng(1)
    pre = rng.integers(1, n + 1, size=200)
    post = rng.integers(1, n + 1, size=200)
    syn = pa.table(
        {
            "body_pre": pre,
            "body_post": post,
            "weight": rng.integers(1, 12, size=200),
            "consensus_nt": rng.choice(["acetylcholine", "gaba"], size=200),
        }
    )
    ann = pa.table(
        {
            "bodyId": np.arange(1, n + 1),
            "cell_type": rng.choice(["exc", "inh"], size=n),
            "region": np.full(n, "SEZ"),
        }
    )
    syn_path = tmp_path / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    ann_path = tmp_path / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    feather.write_feather(syn, syn_path)
    feather.write_feather(ann, ann_path)
    g = load_connectome(syn_path, annotations_path=ann_path)
    assert g.n_nodes > 0
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    tables = [fit for _ in g.type_names]
    state, _ = run_rate(g, tables, n_steps=1, I_ext=np.zeros(g.n_nodes))
    assert state.r.shape == (g.n_nodes,)
