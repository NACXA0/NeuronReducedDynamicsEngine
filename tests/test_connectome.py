"""TC-1.x connectome parser: Feather primary, CSV/Parquet compatible."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from nrde.io.connectome import erdos_renyi_graph, load_alpha_table, load_connectome


def _write_csv(path: Path) -> None:
    path.write_text(
        "pre_id,post_id,synapse_count,nt_type\n"
        "1,2,10,acetylcholine\n"
        "2,3,4,gaba\n"
        "3,1,2,acetylcholine\n",
        encoding="utf-8",
    )


def _write_ann_csv(path: Path) -> None:
    path.write_text(
        "bodyId,cell_type,region\n"
        "1,Kenyon,MB\n"
        "2,MBON,MB\n"
        "3,DAN,MB\n",
        encoding="utf-8",
    )


def test_tc_1_1_csv_and_feather(tmp_path: Path):
    csv_path = tmp_path / "syn.csv"
    _write_csv(csv_path)
    g_csv = load_connectome(csv_path)
    assert g_csv.n_nodes == 3
    assert g_csv.edge_index.shape[1] == 3

    table = pa.table(
        {
            "body_pre": [10, 11],
            "body_post": [11, 10],
            "weight": [5, 7],
            "consensus_nt": ["acetylcholine", "gaba"],
        }
    )
    feat = tmp_path / "syn.feather"
    feather.write_feather(table, feat)
    g_f = load_connectome(feat)
    assert g_f.n_nodes == 2
    assert g_f.edge_weight[1] < 0


def test_tc_1_1_malecns_feather_trio(tmp_path: Path):
    weights = pa.table(
        {
            "body_pre": [1, 2],
            "body_post": [2, 1],
            "weight": [3, 4],
        }
    )
    ann = pa.table({"bodyId": [1, 2], "cell_type": ["T4", "aCC"], "region": ["LOP", "VNC"]})
    nt = pa.table({"bodyId": [1, 2], "consensus_nt": ["acetylcholine", "gaba"]})
    wpath = tmp_path / "connectome-weights-test.feather"
    apath = tmp_path / "body-annotations-test.feather"
    npath = tmp_path / "body-neurotransmitters-test.feather"
    feather.write_feather(weights, wpath)
    feather.write_feather(ann, apath)
    feather.write_feather(nt, npath)
    g = load_connectome(wpath, annotations_path=apath, nt_path=npath)
    assert set(g.type_names) == {"T4", "aCC"}
    assert g.n_nodes == 2
    assert g.edge_weight[1] < 0


def test_tc_1_1_parquet(tmp_path: Path):
    table = pa.table({"pre_id": [1], "post_id": [2], "synapse_count": [3], "nt_type": ["gaba"]})
    path = tmp_path / "s.parquet"
    pq.write_table(table, path)
    g = load_connectome(path)
    assert g.n_nodes == 2
    assert g.edge_weight[0] < 0


def test_tc_1_2_type_grouping(tmp_path: Path):
    syn = tmp_path / "syn.csv"
    ann = tmp_path / "ann.csv"
    _write_csv(syn)
    _write_ann_csv(ann)
    g = load_connectome(syn, annotations_path=ann)
    assert set(g.type_names) == {"Kenyon", "MBON", "DAN"}
    assert g.node_type.shape == (3,)


def test_include_ids_keeps_induced_edges(tmp_path: Path):
    syn = tmp_path / "syn.csv"
    _write_csv(syn)
    g = load_connectome(syn, include_ids=[1, 2])
    assert g.n_nodes == 2
    assert g.edge_index.shape[1] == 1
    assert set(g.node_ids.tolist()) == {1, 2}


def test_tc_1_4_subgraph_by_type(tmp_path: Path):
    syn = tmp_path / "syn.csv"
    ann = tmp_path / "ann.csv"
    _write_csv(syn)
    _write_ann_csv(ann)
    g = load_connectome(syn, annotations_path=ann, include_types=["Kenyon", "MBON"])
    names = set(g.type_names)
    assert "DAN" not in names
    assert g.n_nodes == 2


def test_tc_1_5_alpha_override(tmp_path: Path):
    syn = tmp_path / "syn.csv"
    _write_csv(syn)
    table = load_alpha_table()
    table["acetylcholine|unknown"] = 0.5
    g = load_connectome(syn, alpha_table=table)
    assert g.edge_weight.max() > 1.0


def test_erdos_renyi_two_types():
    g = erdos_renyi_graph(50, p=0.1, n_types=2, seed=1)
    assert g.n_nodes == 50
    assert len(g.type_names) == 2
    assert g.edge_index.shape[0] == 2
