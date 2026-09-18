"""Data preset fetch via manifest (RFC-002 D20): nrde fetch fly."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = _ROOT / "configs" / "manifests" / "fly.yaml"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULT_MANIFEST
    if not p.is_absolute():
        p = _ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def fetch_preset(
    name: str = "fly",
    *,
    tier: str = "seed",
    refresh: bool = False,
    manifest_path: str | Path | None = None,
    skip_weights: bool = True,
) -> dict[str, Any]:
    """Fetch/verify a named data preset. Returns a JSON-serializable report."""
    del name  # only 'fly' for now; manifest embeds preset id
    raw = load_manifest(manifest_path)
    tiers = raw.get("tiers") or {}
    if tier not in tiers:
        raise KeyError(f"Unknown tier {tier!r}. Known: {sorted(tiers)}")
    spec = tiers[tier]
    report: dict[str, Any] = {
        "preset": raw.get("preset", "fly"),
        "tier": tier,
        "version": raw.get("version"),
        "artifacts": [],
        "connectome": None,
        "ok": True,
    }

    for item in spec.get("artifacts") or []:
        rel = str(item["path"])
        path = _ROOT / rel
        entry: dict[str, Any] = {"path": rel, "exists": path.exists()}
        expected = item.get("sha256")
        if not path.exists():
            if tier == "smoke":
                entry["status"] = "missing"
                report["ok"] = False
            else:
                entry["status"] = "missing_seed"
                report["ok"] = False
                entry["hint"] = (
                    f"Seed {rel} not in tree; run nrde offline fit or check out tracked artifacts"
                )
        else:
            digest = _sha256(path)
            entry["sha256"] = digest
            if expected and digest != expected:
                entry["status"] = "hash_mismatch"
                entry["expected"] = expected
                report["ok"] = False
            else:
                entry["status"] = "ok" if not refresh else "ok_refresh_noop"
        report["artifacts"].append(entry)

    conn = spec.get("connectome")
    if conn:
        out_dir = _ROOT / str(conn.get("out_dir", "data/malecns"))
        script = _ROOT / str(conn.get("script", "scripts/download_malecns.py"))
        args = list(conn.get("args") or [])
        if skip_weights and "--skip-weights" not in args:
            args.append("--skip-weights")
        if refresh:
            args.append("--force")
        cmd = [sys.executable, str(script), "--out", str(out_dir), *args]
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            report["connectome"] = {
                "cmd": cmd,
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-500:],
                "stderr_tail": (proc.stderr or "")[-500:],
            }
            if proc.returncode != 0:
                report["ok"] = False
        except OSError as exc:
            report["connectome"] = {"error": str(exc)}
            report["ok"] = False

    report_path = _ROOT / "data" / "fetch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
