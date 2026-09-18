"""CLI smoke tests (IF-4)."""

from __future__ import annotations

from pathlib import Path

from fre.cli import main


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
