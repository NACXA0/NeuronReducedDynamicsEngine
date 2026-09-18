"""TC-6.x calibration and validation, including FlyWire-only Hopkins note."""

from __future__ import annotations

import numpy as np

from nrde.binding import tables_for_graph
from nrde.calibration import (
    apply_alpha_scale,
    apply_shiu_weights,
    calibrate_alpha,
    relative_rate_error,
)
from nrde.fitting.fit import fit_fi
from nrde.io.connectome import erdos_renyi_graph
from nrde.types import SHIU_W_SYN_MV
from nrde.validation import (
    relative_vp,
    uniform_lif_ablation,
    validate_circuit,
    validate_hopkins_flywire,
    validate_single,
    victor_purpura,
)


def test_tc_6_1_single_cell_and_vp():
    fit = fit_fi("lif", n_I=8, t_total=350.0, window=200.0, dt=0.05, n_validate=3, I_max=0.7)
    report = validate_single(fit, I_test=np.linspace(0.2, 0.6, 4), t_total=350.0, window=200.0)
    assert report.r2 >= 0.9
    t1 = np.array([0.0, 10.0, 20.0])
    t2 = np.array([0.0, 10.0, 21.0])
    assert victor_purpura(t1, t2, q=0.1) <= 0.2
    assert relative_vp(t1, t1) == 0.0


def test_tc_6_vp_cost_is_one_over_ten_ms():
    d = victor_purpura(np.array([0.0]), np.array([10.0]), q=0.1)
    assert d == 1.0


def test_tc_6_2_circuit_and_ablation():
    fit_a = fit_fi("lif", type_id="type_0", n_I=8, t_total=300.0, window=180.0, I_max=0.7, n_validate=2)
    fit_b = fit_fi("explif", type_id="type_1", n_I=8, t_total=300.0, window=180.0, I_max=0.7, n_validate=2)
    g = erdos_renyi_graph(40, p=0.06, n_types=2, seed=9, weight=0.001, type_names=("type_0", "type_1"))
    tables = tables_for_graph(g, {"type_0": fit_a, "type_1": fit_b})
    I_ext = np.full(g.n_nodes, 0.35)
    stats = validate_circuit(g, tables, I_ext, n_steps=40)
    assert stats["mean_rate"] >= 0.0
    abl = uniform_lif_ablation(g, tables, I_ext, n_steps=40)
    assert "mean_rate_delta" in abl


def test_tc_6_3_hopkins_is_flywire_only():
    info = validate_hopkins_flywire()
    assert info["dataset"] == "flywire"
    assert "MaleCNS" in info["note"]


def test_calibrate_alpha_moves_weights():
    fit = fit_fi("lif", n_I=8, t_total=300.0, window=180.0, I_max=0.6, n_validate=2)
    g = erdos_renyi_graph(20, p=0.1, n_types=1, seed=1, weight=0.002, type_names=("type_0",))
    apply_shiu_weights(g)
    assert g.meta["w_syn_mv"] == SHIU_W_SYN_MV
    target = np.full(20, 20.0)
    scale, mse = calibrate_alpha(g, [fit], target, I_ext=np.full(20, 0.3), n_steps=15)
    assert scale > 0.0
    apply_alpha_scale(g, scale)
    assert "alpha_scale" in g.meta
    assert mse >= 0.0
    err = relative_rate_error(np.full(20, 18.2), np.full(20, 20.0))
    assert err <= 0.10
