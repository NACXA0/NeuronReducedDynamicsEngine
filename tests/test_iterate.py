"""Auto layer climb (--auto) and one-shot --layers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from nrde.cli import main
from nrde.fitting.iterate import fit_iterative, fit_layers


def _stub_lut(*_args, **_kwargs):
    axis = np.linspace(0.0, 0.8, 4)
    return {
        "lut_I": axis,
        "lut_dt": np.linspace(0.0, 10.0, 4),
        "lut_spike": np.zeros((4, 4), dtype=np.float32),
        "lut_V": np.zeros((4, 4), dtype=np.float32),
        "lut_hit_rate": 0.0,
        "lut_sched_I": axis,
        "lut_sched_t": np.full((4, 4), np.nan),
    }


def _passed(fit):
    return SimpleNamespace(r2=fit.r2, relative_vp=0.01, nfr5_pass=True, nfr4_pass=True)


def test_lif_auto_stops_at_l0(monkeypatch):
    def _no_lut(*_args, **_kwargs):
        raise AssertionError("L0 ceiling must not attach a LUT")

    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _no_lut)
    fit, rounds = fit_iterative(
        "lif",
        type_id="lif_auto_l0",
        I_min=0.0,
        I_max=0.8,
        n_I=16,
        t_total=400.0,
        window=250.0,
        n_validate=2,
        max_layer="L0",
    )
    assert fit.quality == "ok"
    assert fit.lut_sched_t is None
    assert fit.srm is None
    assert fit.fallback_ode is False
    assert fit.method == "analytic"
    assert rounds[0]["action"] == "fit_fi"
    assert all(row["action"] != "densify" for row in rounds)
    assert "prefer=efficiency" in fit.notes


def _vp_script(values: list[float]):
    cursor = {"i": 0}

    def _validate(fit):
        vp = values[min(cursor["i"], len(values) - 1)]
        cursor["i"] += 1
        return SimpleNamespace(r2=fit.r2, relative_vp=vp, nfr5_pass=True, nfr4_pass=True)

    return _validate


def test_efficiency_keeps_the_first_pass(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    monkeypatch.setattr("nrde.fitting.iterate.validate_single", _vp_script([0.08]))
    fit, rounds = fit_iterative(
        "adexp",
        type_id="adexp_eff",
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
        max_layer="L2",
        prefer="efficiency",
    )
    assert fit.srm is None
    assert fit.quality == "ok"
    assert not any(row["action"] == "attach_srm" for row in rounds)


def test_neutral_reverts_when_vp_does_not_drop(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    monkeypatch.setattr("nrde.fitting.iterate.validate_single", _vp_script([0.08, 0.08]))
    fit, rounds = fit_iterative(
        "adexp",
        type_id="adexp_neutral_revert",
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
        max_layer="L2",
        prefer="neutral",
    )
    assert fit.srm is None
    assert fit.layer == "L1"
    assert any(row["action"] == "revert_srm" for row in rounds)


def test_neutral_keeps_srm_when_vp_drops(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    monkeypatch.setattr("nrde.fitting.iterate.validate_single", _vp_script([0.08, 0.04]))
    fit, _rounds = fit_iterative(
        "adexp",
        type_id="adexp_neutral_keep",
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
        max_layer="L2",
        prefer="neutral",
    )
    assert fit.srm is not None
    assert fit.layer == "L2"


def test_fidelity_climbs_to_the_ceiling(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    monkeypatch.setattr("nrde.fitting.iterate.validate_single", _passed)
    fit, rounds = fit_iterative(
        "lif",
        type_id="lif_fidelity",
        n_I=16,
        t_total=400.0,
        window=250.0,
        n_validate=2,
        max_layer="L2",
        prefer="fidelity",
    )
    assert fit.lut_sched_t is not None
    assert fit.srm is not None
    assert fit.quality == "ok"
    assert any(row["action"] == "attach_srm" for row in rounds)
    assert "prefer=fidelity" in fit.notes


def test_unknown_prefer_rejected():
    import pytest

    with pytest.raises(ValueError, match="prefer"):
        fit_iterative("lif", prefer="fast", max_layer="L0", n_I=4, n_validate=2)


def test_adexp_max_l1_attaches_lut(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    monkeypatch.setattr("nrde.fitting.iterate.validate_single", _passed)
    fit, rounds = fit_iterative(
        "adexp",
        type_id="adexp_auto_l1",
        I_min=0.0,
        I_max=0.8,
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
        max_layer="L1",
    )
    assert fit.lut_sched_t is not None
    assert fit.srm is None
    assert any(row["action"] == "attach_lut" for row in rounds)


def test_tight_vp_gate_mixes_l1l2_and_falls_back(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    monkeypatch.setattr("nrde.validation.REL_VP_GATE", 0.0)
    fit, rounds = fit_iterative(
        "adexp",
        type_id="adexp_auto_l2",
        I_min=0.0,
        I_max=0.8,
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
        max_layer="L2",
    )
    assert fit.lut_sched_t is not None
    assert fit.srm is not None or fit.fallback_ode
    assert fit.fallback_ode is True
    assert fit.quality == "poor"
    assert any(row["action"] == "attach_srm" for row in rounds)
    assert any(row["action"] == "fallback_ode" for row in rounds)


def test_max_layer_l3_records_deferral(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    monkeypatch.setattr("nrde.validation.REL_VP_GATE", 0.0)
    fit, rounds = fit_iterative(
        "adexp",
        type_id="adexp_auto_l3",
        I_min=0.0,
        I_max=0.8,
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
        max_layer="L3",
    )
    assert "l3=deferred_v0.2" in fit.notes
    assert any(row["action"] == "l3_deferred" for row in rounds)
    assert fit.fallback_ode is True


def test_layers_l1l2_does_not_densify(monkeypatch):
    monkeypatch.setattr("nrde.fitting.iterate.fit_spike_lut", _stub_lut)
    fit, rounds = fit_layers(
        "L1+L2",
        "lif",
        type_id="lif_fixed",
        I_min=0.0,
        I_max=0.8,
        n_I=8,
        t_total=200.0,
        window=120.0,
        n_validate=2,
    )
    assert [row["action"] for row in rounds] == ["fit_fi", "attach_lut", "attach_srm"]
    assert {row["n_I"] for row in rounds} == {8}
    assert fit.lut_sched_t is not None
    assert fit.srm is not None
    assert fit.fallback_ode is False


def test_cli_auto_json_and_one_shot_unchanged(tmp_path, capsys):
    out = tmp_path / "auto.npz"
    rc = main(
        [
            "offline",
            "fit",
            "--auto",
            "--max-layer",
            "L0",
            "--model",
            "lif",
            "--type-id",
            "lif_auto_cli",
            "--n-I",
            "16",
            "--t-total",
            "400",
            "--window",
            "250",
            "--I-max",
            "0.8",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fallback_ode"] is False
    assert payload["prefer"] == "efficiency"
    assert payload["rounds"]
    assert out.exists()

    one = tmp_path / "once.npz"
    rc = main(
        [
            "fit",
            "--model",
            "lif",
            "--n-I",
            "6",
            "--t-total",
            "200",
            "--window",
            "120",
            "--I-max",
            "0.5",
            "--out",
            str(one),
        ]
    )
    assert rc == 0
    once_payload = json.loads(capsys.readouterr().out)
    assert "rounds" not in once_payload
    assert "fallback_ode" not in once_payload
    assert "prefer" not in once_payload
