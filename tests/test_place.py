"""Identity-axis place correction: identity is a no-op; region gain is local."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from nrde.binding import load_type_map
from nrde.engine.rate import init_rate_state, step_rate
from nrde.fitting.fit import _config_hash, load_fit, save_fit
from nrde.io.connectome import load_connectome
from nrde.types import NPZ_SCHEMA_VERSION, FittedActivation, GraphData, PlaceMod


def _fit(**kwargs) -> FittedActivation:
    fit = FittedActivation(
        schema_version=NPZ_SCHEMA_VERSION,
        type_id="t",
        model="lif",
        params={},
        method="linear",
        I_grid=np.array([0.0, 1.0]),
        r_grid=np.array([0.0, 10.0]),
        I_onset=0.0,
        r2=1.0,
        mse=0.0,
        quality="ok",
    )
    return replace(fit, **kwargs) if kwargs else fit


def _graph(region: np.ndarray | None, xyz: np.ndarray | None) -> GraphData:
    return GraphData(
        n_nodes=2,
        node_ids=np.array([1, 2], dtype=np.int64),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=np.zeros(0, dtype=np.float32),
        node_type=np.zeros(2, dtype=np.int32),
        type_names=("t",),
        region=region,
        xyz=xyz,
    )


def _step(fit: FittedActivation, graph: GraphData) -> np.ndarray:
    state = step_rate(init_rate_state(2), graph, [fit], np.array([0.5, 0.5]))
    return state.r


def test_identity_place_matches_type_table_alone():
    fit = _fit()
    region = np.array(["MB", "AL"], dtype=object)
    bare = _step(fit, _graph(region, None))
    zeros = _step(fit, _graph(region, np.zeros((2, 3))))
    missing = _step(fit, _graph(region, np.full((2, 3), np.nan)))
    np.testing.assert_allclose(bare, [5.0, 5.0])
    np.testing.assert_allclose(zeros, bare)
    np.testing.assert_allclose(missing, bare)


def test_region_gain_touches_only_that_region():
    fit = _fit(place=PlaceMod(region_gain={"MB": 0.5}))
    region = np.array(["MB", "AL"], dtype=object)
    rates = _step(fit, _graph(region, None))
    np.testing.assert_allclose(rates, [7.5, 5.0])


def test_unknown_region_and_nan_xyz_add_nothing():
    fit = _fit(place=PlaceMod(region_gain={"MB": 0.5}, xyz_gain=np.array([0.2, 0.0, 0.0])))
    region = np.array(["ZZ", "ZZ"], dtype=object)
    xyz = np.array([[np.nan, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    rates = _step(fit, _graph(region, xyz))
    np.testing.assert_allclose(rates, [5.0, 5.0])
    assert np.isfinite(rates).all()


def test_old_npz_loads_identity_builtin(tmp_path: Path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        schema_version=np.array([NPZ_SCHEMA_VERSION]),
        I_grid=np.array([0.0, 1.0]),
        r_grid=np.array([0.0, 10.0]),
        I_onset=np.array([0.0]),
        r2=np.array([1.0]),
        mse=np.array([0.0]),
        layer=np.array(["L0"]),
    )
    loaded = load_fit(path)
    assert loaded.place.is_identity()
    assert loaded.teacher is None


def test_place_does_not_change_config_hash(tmp_path: Path):
    plain = _fit()
    moved = _fit(place=PlaceMod(region_gain={"MB": 0.25}))
    assert _config_hash(plain) == _config_hash(moved)
    save_fit(moved, tmp_path / "moved.npz", skip_if_unchanged=False)
    loaded = load_fit(tmp_path / "moved.npz")
    assert loaded.place.region_gain["MB"] == 0.25
    assert loaded.config_hash == _config_hash(plain)


def test_type_map_teacher_slot():
    by_id = {entry.type_id: entry for entry in load_type_map("configs/type_model_map.yaml")}
    t4 = by_id["T4"].teacher
    acc = by_id["aCC"].teacher
    assert t4 is not None and acc is not None
    assert t4.kind == "builtin" and t4.ref == "adexp"
    assert "ModelDB" in t4.notes
    assert acc.kind == "builtin" and "ModelDB" in acc.notes
    assert by_id["hh_demo"].teacher is not None
    assert by_id["hh_demo"].teacher.kind == "builtin"


def test_annotation_xyz_or_unset(tmp_path: Path):
    syn = tmp_path / "syn.csv"
    syn.write_text("pre_id,post_id,synapse_count\n1,2,1\n", encoding="utf-8")
    bare = tmp_path / "ann.csv"
    bare.write_text("bodyId,cell_type,region\n1,T4,LOP\n2,T4,LOP\n", encoding="utf-8")
    g = load_connectome(syn, annotations_path=bare)
    assert g.xyz is None
    assert g.meta["xyz_source"] == "unset"
    located = tmp_path / "ann_xyz.csv"
    located.write_text(
        "bodyId,cell_type,region,soma_x,soma_y,soma_z\n1,T4,LOP,1,2,3\n2,T4,LOP,4,5,6\n",
        encoding="utf-8",
    )
    g2 = load_connectome(syn, annotations_path=located)
    assert g2.meta["xyz_source"] == "annotation"
    assert g2.xyz is not None
    np.testing.assert_allclose(g2.xyz, [[1, 2, 3], [4, 5, 6]])
