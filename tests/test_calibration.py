"""TC-6 calibration: Shiu W_syn start and relative rate error (RFC-001 D2)."""

from __future__ import annotations

import numpy as np

from nrde.calibration import (
    apply_alpha_scale,
    apply_shiu_weights,
    calibrate_alpha,
    relative_rate_error,
    shiu_alpha_na,
)
from nrde.engine.rate import run_rate
from nrde.fitting.fit import fit_fi
from nrde.io.connectome import erdos_renyi_graph
from nrde.types import SHIU_ALPHA_NA, SHIU_W_SYN_MV


def test_shiu_alpha_matches_published_w_syn():
    assert SHIU_W_SYN_MV == 0.275
    assert shiu_alpha_na() == SHIU_ALPHA_NA
    assert abs(SHIU_ALPHA_NA - 0.0275) < 1e-12


def test_calibrate_alpha_from_shiu_start():
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, I_max=0.6, n_validate=2)
    g = erdos_renyi_graph(16, p=0.1, n_types=1, seed=3, weight=0.002, type_names=("type_0",))
    apply_shiu_weights(g)
    assert g.meta["w_syn_mv"] == SHIU_W_SYN_MV
    scale, mse = calibrate_alpha(g, [fit], np.full(16, 15.0), I_ext=np.full(16, 0.3), n_steps=12)
    apply_alpha_scale(g, scale)
    assert scale > 0.0
    assert mse >= 0.0
    assert relative_rate_error(np.full(16, 14.0), np.full(16, 15.0)) <= 0.10


def test_shiu_circuit_recovers_rate_within_10_percent():
    """RFC-001 §4.4: from Shiu weights, a reachable target stays within 10%."""
    fit = fit_fi("lif", n_I=16, t_total=300.0, window=180.0, I_max=0.8, n_validate=2)
    n = 32
    I_ext = np.full(n, 0.24)
    held_out = 3.0

    def circuit():
        g = erdos_renyi_graph(n, p=0.2, n_types=1, seed=2, weight=1.0, type_names=("lif",))
        apply_shiu_weights(g, synapse_counts=np.full(g.edge_weight.shape, 1.0))
        assert g.meta["w_syn_mv"] == SHIU_W_SYN_MV
        return g

    observed = circuit()
    apply_alpha_scale(observed, held_out)
    target, _ = run_rate(observed, [fit], n_steps=30, I_ext=I_ext, record_trace=False)
    start = circuit()
    scale, _mse = calibrate_alpha(start, [fit], target.r, I_ext=I_ext, n_steps=30)
    apply_alpha_scale(start, scale)
    got, _ = run_rate(start, [fit], n_steps=30, I_ext=I_ext, record_trace=False)
    assert relative_rate_error(got.r, target.r) <= 0.10
