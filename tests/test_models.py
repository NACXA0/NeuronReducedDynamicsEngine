"""TC-2.x model registry and LIF analytic checks."""

from __future__ import annotations

import numpy as np
import pytest

from fre.models import get_model, list_models
from fre.models.lif import LIFParams, analytic_rate
from fre.sim import firing_rate, simulate_spikes


def test_tc_2_1_builtin_models_registered():
    names = list_models()
    for required in ("lif", "explif", "adexp", "hh", "izhikevich"):
        assert required in names
        get_model(required)


def test_tc_2_2_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("not-a-model")


def test_tc_2_3_lif_analytic_step_response():
    p = LIFParams()
    I_rh = p.gL * (p.Vth - p.EL)
    assert analytic_rate(0.0, p) == 0.0
    assert analytic_rate(I_rh, p) == 0.0
    rate = analytic_rate(0.5, p)
    assert rate > 10.0
    num = firing_rate("lif", {}, I=0.5, t_total=400.0, dt=0.05, window=250.0)
    assert abs(num - rate) / rate < 0.15


def test_tc_2_4_explif_and_hh_spike():
    explif_spikes = simulate_spikes("explif", {}, I=0.4, t_total=300.0, dt=0.05)
    assert explif_spikes.size >= 1
    hh_spikes = simulate_spikes("hh", {}, I=1.0, t_total=200.0, dt=0.02)
    assert hh_spikes.size >= 1


def test_tc_2_5_adexp_reset_increments_w():
    spec = get_model("adexp")
    p = spec.Params()
    y = spec.y0(p)
    y[0] = p.Vth + 1.0
    y2 = spec.reset(y, p)
    assert y2[0] == p.Vreset
    assert y2[1] == pytest.approx(y[1] + p.b)


def test_tc_2_6_izhikevich_fires():
    spikes = simulate_spikes("izhikevich", {}, I=10.0, t_total=200.0, dt=0.5)
    assert spikes.size >= 1
    assert np.all(np.diff(spikes) > 0)
