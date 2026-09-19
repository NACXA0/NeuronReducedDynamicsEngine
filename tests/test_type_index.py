"""Type-index construction and progress output."""

from __future__ import annotations

import numpy as np

from nrde.engine.rate import batched_apply, coalesce_type_tables
from nrde.engine.sparse import indices_for_codes
from nrde.fitting.fit import fit_fi
from nrde.progress import get_progress, progress_session


def test_indices_match_per_code_scan():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 5, size=2000)
    got = indices_for_codes(codes, 7)
    assert len(got) == 7
    for t in range(7):
        np.testing.assert_array_equal(got[t], np.flatnonzero(codes == t))


def test_indices_ignore_out_of_range_codes():
    codes = np.array([-1, 0, 3, 9, 1, 0], dtype=np.int32)
    got = indices_for_codes(codes, 4)
    for t in range(4):
        np.testing.assert_array_equal(got[t], np.flatnonzero(codes == t))


def test_coalesce_shares_one_table_and_matches_per_type_eval():
    fit_a = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.6)
    fit_b = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5, I_min=0.05)
    codes = np.array([0, 1, 0, 2, 1, 2, 0], dtype=np.int32)
    tables = [fit_a, fit_a, fit_b]
    indices = indices_for_codes(codes, 3)
    groups = coalesce_type_tables(tables, indices)
    assert len(groups) == 2
    assert groups[0][0] is fit_a
    assert groups[1][0] is fit_b
    I = np.linspace(0.0, 0.4, codes.size)
    got, oor = batched_apply(tables, I, codes, type_indices=indices)
    ref = np.zeros_like(I)
    ref_oor = 0
    for table, idx in zip(tables, indices, strict=True):
        if idx.size == 0:
            continue
        rt, n = table.eval_rate(I[idx])
        ref[idx] = rt
        ref_oor += n
    np.testing.assert_allclose(got, ref)
    assert oor == ref_oor


def test_progress_session_writes_stages_and_bar(capsys):
    with progress_session(enabled=True):
        progress = get_progress()
        progress.start("reading synapses")
        progress.done("3 edges")
        progress.tick(1, 2, "rate steps")
        progress.tick(2, 2, "rate steps")
    err = capsys.readouterr().err
    assert "reading synapses" in err
    assert "3 edges" in err
    assert "rate steps" in err
    assert "#" in err


def test_progress_disabled_is_silent(capsys):
    with progress_session(enabled=False):
        progress = get_progress()
        progress.start("reading synapses")
        progress.done("3 edges")
        progress.tick(1, 1, "rate steps")
    assert capsys.readouterr().err == ""


def test_chunked_type_index_matches_scan():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 6, size=500)
    with progress_session(enabled=True) as progress:
        progress.chunk = 40
        got = indices_for_codes(codes, 9)
    for t in range(9):
        np.testing.assert_array_equal(got[t], np.flatnonzero(codes == t))


def test_chunked_csr_matches_scipy_matvec():
    from scipy.sparse import csr_matrix

    from nrde.engine.sparse import build_csr

    rng = np.random.default_rng(2)
    n_nodes = 15
    pre = rng.integers(0, n_nodes, size=80)
    post = rng.integers(0, n_nodes, size=80)
    pre = np.concatenate([pre, pre[:12]])
    post = np.concatenate([post, post[:12]])
    weight = rng.random(pre.size).astype(np.float32)
    ref = csr_matrix((weight, (post, pre)), shape=(n_nodes, n_nodes), dtype=np.float32)
    with progress_session(enabled=True) as progress:
        progress.chunk = 9
        got = build_csr(np.vstack([pre, post]), weight, n_nodes)
    x = rng.random(n_nodes).astype(np.float32)
    np.testing.assert_allclose(got @ x, ref @ x, rtol=1e-5, atol=1e-5)


def test_progress_load_matches_quiet_path(tmp_path, capsys):
    import pyarrow as pa
    import pyarrow.feather as feather

    from nrde.io.connectome import load_connectome

    rng = np.random.default_rng(3)
    n_edges = 48
    pre = rng.integers(1, 18, size=n_edges)
    post = rng.integers(1, 18, size=n_edges)
    syn = pa.table({"body_pre": pre, "body_post": post, "weight": rng.integers(1, 5, size=n_edges)})
    ann = pa.table(
        {
            "bodyId": np.arange(1, 18),
            "cell_type": np.array(["Kenyon", "MBON", "DAN"] * 6)[:17],
            "region": np.array(["MB", "AL", ""] * 6)[:17],
        }
    )
    syn_path = tmp_path / "syn.feather"
    ann_path = tmp_path / "ann.feather"
    feather.write_feather(syn, syn_path, chunksize=10)
    feather.write_feather(ann, ann_path, chunksize=5)
    quiet = load_connectome(syn_path, annotations_path=ann_path)
    with progress_session(enabled=True) as progress:
        progress.chunk = 7
        shown = load_connectome(syn_path, annotations_path=ann_path)
    err = capsys.readouterr().err
    assert "edges" in err and "nodes" in err and "#" in err
    assert quiet.n_nodes == shown.n_nodes
    assert quiet.type_names == shown.type_names
    np.testing.assert_array_equal(quiet.node_type, shown.node_type)
    np.testing.assert_array_equal(quiet.edge_index, shown.edge_index)
    np.testing.assert_array_equal(quiet.edge_weight, shown.edge_weight)
    np.testing.assert_array_equal(quiet.region, shown.region)
