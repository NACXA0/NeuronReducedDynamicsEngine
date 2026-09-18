"""TC-3.x offline fitting, serialization, and non-LIF gate (FR-3.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fre.offline.fit import attach_lut, fit_fi, fit_spike_lut, load_fit, save_fit
from fre.types import NPZ_SCHEMA_VERSION


def _fast_kwargs(**extra):
    kw = dict(n_I=10, t_total=400.0, window=250.0, dt=0.05, n_validate=4)
    kw.update(extra)
    return kw


def test_tc_3_1_explif_fi_meets_nfr4():
    fit = fit_fi("explif", type_id="exc", I_min=0.0, I_max=0.7, **_fast_kwargs())
    assert fit.r2 >= 0.98
    assert fit.quality == "ok"
    assert fit.r_grid.max() > 5.0


def test_tc_3_6_hh_or_adexp_fi_nfr4():
    hh = fit_fi(
        "hh",
        type_id="hh_demo",
        I_min=0.4,
        I_max=1.6,
        n_I=8,
        t_total=250.0,
        window=150.0,
        dt=0.02,
        n_validate=3,
    )
    adexp = fit_fi(
        "adexp",
        type_id="adexp_demo",
        I_min=0.15,
        I_max=0.8,
        **_fast_kwargs(n_I=8, n_validate=3),
    )
    assert hh.model == "hh"
    assert "PCHIP" in hh.notes or hh.method == "pchip"
    assert adexp.r_grid.max() > 0.0
    assert max(hh.r2, adexp.r2) >= 0.90
    assert adexp.r2 >= 0.98 or hh.r2 >= 0.98


def test_tc_3_3_roundtrip_npz(tmp_path: Path):
    fit = fit_fi("lif", type_id="lif0", I_min=0.0, I_max=0.6, **_fast_kwargs())
    lut = fit_spike_lut("lif", I_min=0.0, I_max=0.6, n_I=8, n_dt=8, dt=1.0)
    fit = attach_lut(fit, lut)
    path = save_fit(fit, tmp_path / "fit.npz")
    loaded = load_fit(path)
    assert loaded.schema_version >= NPZ_SCHEMA_VERSION - 1
    assert loaded.type_id == "lif0"
    assert loaded.lut_I is not None
    r, oor = loaded.eval_rate(np.array([-1.0, 0.3, 9.0]))
    assert oor == 2
    assert np.all(r >= 0.0)


def test_tc_3_5_saturation_extrapolation():
    fit = fit_fi("lif", I_min=0.0, I_max=0.5, **_fast_kwargs())
    r, oor = fit.eval_rate(np.array([10.0]))
    assert oor == 1
    assert r[0] == fit.r_grid[-1] or r[0] <= fit.r_grid.max()


def test_tc_3_4_parallel_scan_matches_serial():
    a = fit_fi("lif", I_min=0.2, I_max=0.5, n_jobs=1, **_fast_kwargs(n_I=6, n_validate=2))
    b = fit_fi("lif", I_min=0.2, I_max=0.5, n_jobs=2, **_fast_kwargs(n_I=6, n_validate=2))
    np.testing.assert_allclose(a.r_grid, b.r_grid, rtol=1e-6, atol=1e-6)
