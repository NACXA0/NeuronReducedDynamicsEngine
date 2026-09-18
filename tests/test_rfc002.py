"""RFC-002 surface: public API, presets, ModelSpec, config_hash idempotency."""

from __future__ import annotations

from pathlib import Path

import nrde
from nrde.cli import main
from nrde.fitting.fit import fit_fi, load_fit, save_fit
from nrde.specs import ModelSpec, model_spec_from_registry


def test_public_api_surface():
    assert set(nrde.__all__) == {"ModelSpec", "demo", "make", "offline", "run", "__version__"}
    assert "FittedActivation" not in nrde.__all__
    assert "GraphData" not in nrde.__all__
    assert not hasattr(nrde, "FittedActivation")
    # Callables are the product surface.
    assert callable(nrde.make) and callable(nrde.demo) and callable(nrde.run) and callable(nrde.offline)


def test_model_spec_from_builtin():
    spec = model_spec_from_registry("lif")
    assert isinstance(spec, ModelSpec)
    assert spec.protocol_version >= 1
    assert spec.state_dims >= 1


def test_config_hash_idempotent_save(tmp_path: Path):
    fit = fit_fi(
        "lif",
        type_id="idem",
        I_min=0.0,
        I_max=0.5,
        n_I=6,
        t_total=250.0,
        window=150.0,
        dt=0.05,
        n_validate=2,
    )
    path = tmp_path / "idem.npz"
    save_fit(fit, path)
    mtime1 = path.stat().st_mtime_ns
    meta1 = (tmp_path / "idem.meta.json").read_text(encoding="utf-8")
    save_fit(fit, path, skip_if_unchanged=True)
    mtime2 = path.stat().st_mtime_ns
    meta2 = (tmp_path / "idem.meta.json").read_text(encoding="utf-8")
    assert mtime1 == mtime2
    assert meta1 == meta2
    loaded = load_fit(path)
    assert loaded.config_hash == fit.config_hash


def test_cli_demo_headless():
    rc = main(["demo", "flygym", "--steps", "20", "--headless"])
    assert rc == 0


def test_cli_offline_fit_alias(tmp_path: Path):
    out = tmp_path / "f.npz"
    rc = main(
        [
            "offline",
            "fit",
            "--model",
            "lif",
            "--n-I",
            "6",
            "--t-total",
            "250",
            "--window",
            "150",
            "--I-max",
            "0.5",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_make_preset():
    env = nrde.make("flygym-demo-v01", steps=10, headless=True)
    result = env.run(10)
    assert result["n_steps"] == 10
    assert result["finite"]
