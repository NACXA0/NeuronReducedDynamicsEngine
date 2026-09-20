"""CLI smoke tests (IF-4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nrde.cli import main
from nrde.fitting.fit import fit_fi, save_fit


def _tiny_fit(tmp_path: Path, type_id: str, model: str = "lif") -> Path:
    fit = fit_fi(
        model,
        type_id=type_id,
        n_I=6,
        t_total=200.0,
        window=120.0,
        I_max=0.5,
        n_validate=2,
    )
    path = tmp_path / f"{type_id}.npz"
    save_fit(fit, path, skip_if_unchanged=False)
    return path


def test_cli_fit_and_validate(tmp_path: Path):
    out = tmp_path / "f.npz"
    rc = main(
        [
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
    rc2 = main(["validate", "--fit", str(out)])
    assert rc2 in (0, 1)
    rc3 = main(["validate", "--hopkins"])
    assert rc3 == 0
    rc4 = main(["validate", "--pysr-exam"])
    assert rc4 == 0


def test_cli_simulate_single_fit_fallback(tmp_path: Path, capsys):
    path = _tiny_fit(tmp_path, "aCC")
    rc = main(["run", "simulate", "--fit", str(path), "--n", "20", "--steps", "5"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["n_nodes"] == 20
    assert payload["bound"] == {"type_0": "aCC", "type_1": "aCC"}
    assert "mean_rate" in payload


def test_cli_simulate_fit_map(tmp_path: Path, capsys):
    prey = _tiny_fit(tmp_path, "prey", model="explif")
    predator = _tiny_fit(tmp_path, "predator", model="lif")
    rc = main(
        [
            "run",
            "simulate",
            "--fit-map",
            f"prey={prey}",
            "--fit-map",
            f"predator={predator}",
            "--type-names",
            "prey,predator",
            "--n",
            "20",
            "--steps",
            "5",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["bound"] == {"prey": "prey", "predator": "predator"}


def test_cli_simulate_missing_map_without_fallback(tmp_path: Path, capsys):
    prey = _tiny_fit(tmp_path, "prey")
    rc = main(
        [
            "run",
            "simulate",
            "--fit-map",
            f"prey={prey}",
            "--type-names",
            "prey,predator",
            "--n",
            "12",
            "--steps",
            "3",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "predator" in err


def test_cli_simulate_requires_fit_or_map(capsys):
    rc = main(["run", "simulate", "--n", "8", "--steps", "2"])
    assert rc == 2
    assert "simulate needs --fit" in capsys.readouterr().err


def test_keyboard_interrupt_exits_clean(monkeypatch, capsys):
    def parse_args(self, args=None, namespace=None):
        del self, args, namespace

        def boom(_args):
            raise KeyboardInterrupt

        return argparse.Namespace(func=boom)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_args)
    assert main([]) == 130
    err = capsys.readouterr().err
    assert "interrupted" in err
    assert "Traceback" not in err
