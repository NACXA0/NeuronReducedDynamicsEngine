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


def test_readme_cites_ledger_numbers():
    ledger = json.loads((ROOT / "docs" / "reports" / "figures" / "evidence.json").read_text(encoding="utf-8"))
    results = ledger["results"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    cited = [
        f"{float(results['r2_F_vs_ode']):.6f}",
        f"{float(results['rmse_F_Hz']):.2f}",
        f"{float(results['rmse_lif_Hz']):.2f}",
        f"{float(results['wall_ms']['nrde_F_circuit']):.2f}",
        f"{float(results['wall_ms']['ode_adexp_scaled']):.1f}",
    ]
    missing = [item for item in cited if item not in readme]
    assert not missing, missing
    assert "docs/reports/evidence.md" in readme
    assert results["nfr4_pass"] is True
    assert ledger["dataset"]["n_artifacts"] == 33
