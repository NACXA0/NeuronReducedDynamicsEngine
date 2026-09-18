from __future__ import annotations

from nrde.binding import load_type_map, tables_for_graph
from nrde.fitting.fit import fit_fi
from nrde.io.connectome import erdos_renyi_graph


def test_load_type_map():
    entries = load_type_map("configs/type_model_map.yaml")
    names = {e.type_id for e in entries}
    assert "default excitatory" in names
    assert "T4" in names and "aCC" in names
    tiers = {e.type_id: e.tier for e in entries}
    assert tiers["T4"] == "literature"
    assert tiers["default excitatory"] == "default"


def test_tables_for_graph_fallback():
    fit = fit_fi("lif", type_id="default excitatory", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    g = erdos_renyi_graph(10, p=0.1, n_types=2, seed=0, type_names=("missing_a", "missing_b"))
    tables = tables_for_graph(g, {"default excitatory": fit}, default_key="default excitatory")
    assert len(tables) == 2
