"""A2 fetch + A1 asset pipeline smoke (RFC-002 v1.1)."""

from __future__ import annotations

from pathlib import Path

from nrde.cli import main
from nrde.fetch import fetch_preset


def test_fetch_smoke_ok():
    report = fetch_preset("fly", tier="smoke")
    assert report["tier"] == "smoke"
    assert report["ok"] is True
    assert all(a["status"] == "ok" for a in report["artifacts"])


def test_cli_fetch_seed():
    rc = main(["fetch", "fly", "--tier", "seed"])
    assert rc == 0


def test_render_demo_script(tmp_path: Path):

    # scripts may not be a package — invoke via path run
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "render_demo.py"
    spec = importlib.util.spec_from_file_location("render_demo", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "demo"
    assert mod.main(["--steps", "5", "--seed", "1", "--out", str(out)]) == 0
    assert (out / "WATERMARK.txt").exists()
    assert (out / "last_run.json").exists()


def test_make_colab_and_release(tmp_path: Path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]

    def _load(name: str):
        p = root / "scripts" / name
        spec = importlib.util.spec_from_file_location(name, p)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    colab = _load("make_colab.py")
    nb = tmp_path / "demo.ipynb"
    assert colab.main(["--out", str(nb)]) == 0
    assert nb.exists() and "nbformat" in nb.read_text(encoding="utf-8")

    rel = _load("make_release.py")
    zpath = tmp_path / "off.zip"
    assert rel.main(["--out", str(zpath)]) == 0
    assert zpath.exists()
