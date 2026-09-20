"""Committed README evidence: hashes and cited R² must match the ledger."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pretrained_tier_has_33_ok_files():
    from nrde.fetch import fetch_preset

    report = fetch_preset("fly", tier="pretrained")
    assert report["ok"] is True
    assert len(report["artifacts"]) == 33
    assert all(a.get("status") in {"ok", "ok_refresh_noop"} for a in report["artifacts"])


def test_evidence_ledger_verify():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_figures.py"), "--verify"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_readme_cites_ledger_r2_and_evidence_doc():
    ledger = json.loads((ROOT / "docs" / "reports" / "figures" / "evidence.json").read_text(encoding="utf-8"))
    r2 = f"{float(ledger['results']['r2_F_vs_ode']):.3f}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert r2 in readme
    assert "docs/reports/evidence.md" in readme
    assert ledger["results"]["nfr4_pass"] is True
    assert ledger["dataset"]["n_artifacts"] == 33
