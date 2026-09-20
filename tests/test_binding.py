from __future__ import annotations

from pathlib import Path

import pytest

from nrde.binding import (
    bind_tables,
    bound_type_ids,
    load_type_map,
    parse_fit_map,
    parse_type_names,
    tables_for_graph,
)
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


def test_parse_fit_map_and_type_names():
    parsed = parse_fit_map(["prey=/tmp/a.npz", "predator=/tmp/b.npz"])
    assert parsed["prey"] == Path("/tmp/a.npz")
    assert parsed["predator"] == Path("/tmp/b.npz")
    assert parse_type_names(None) is None
    assert parse_type_names("prey,predator") == ("prey", "predator")
    with pytest.raises(ValueError, match="TYPE=PATH"):
        parse_fit_map(["prey"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_fit_map(["prey=/tmp/a.npz", "prey=/tmp/c.npz"])
    with pytest.raises(ValueError, match="empty"):
        parse_type_names("prey,")


def test_bind_tables_requires_fallback():
    prey = fit_fi("explif", type_id="prey", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    predator = fit_fi("lif", type_id="predator", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.5)
    g = erdos_renyi_graph(12, p=0.1, n_types=2, seed=0, type_names=("prey", "predator"))
    tables = bind_tables(g, {"prey": prey, "predator": predator})
    assert bound_type_ids(g, tables) == {"prey": "prey", "predator": "predator"}
    with_fallback = bind_tables(g, {"prey": prey}, fallback=predator)
    assert with_fallback[0] is prey
    assert all(t is predator for i, t in enumerate(with_fallback) if g.type_names[i] != "prey")
    with pytest.raises(KeyError, match="predator"):
        bind_tables(g, {"prey": prey})

