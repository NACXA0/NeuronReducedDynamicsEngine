#!/usr/bin/env python3
"""Optional local fits for --prefer efficiency / neutral / fidelity.

Does not touch seed artifacts or datasets/fly/fits. Register with
``nrde fetch fly --tier prefer`` only; the default pretrained tier ignores them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from nrde.fitting.fit import save_fit
from nrde.fitting.iterate import fit_iterative

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "datasets" / "fly" / "prefer"
_JOBS = (
    ("efficiency", "lif", "lif_efficiency"),
    ("efficiency", "adexp", "aCC_efficiency"),
    ("neutral", "adexp", "aCC_neutral"),
    ("fidelity", "adexp", "aCC_fidelity"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for prefer, model, type_id in _JOBS:
        fit, rounds = fit_iterative(
            model,
            type_id=type_id,
            I_min=0.0,
            I_max=0.8,
            n_I=8,
            t_total=300.0,
            window=180.0,
            n_validate=2,
            max_layer="L2",
            prefer=prefer,
        )
        path = save_fit(fit, _OUT / f"{type_id}.npz", skip_if_unchanged=False)
        report[type_id] = {
            "prefer": prefer,
            "model": model,
            "layer": fit.layer,
            "quality": fit.quality,
            "r2": fit.r2,
            "method": fit.method,
            "fallback_ode": bool(fit.fallback_ode),
            "has_lut": fit.lut_sched_t is not None,
            "has_srm": fit.srm is not None,
            "path": str(path.relative_to(_ROOT)),
            "rounds": rounds,
        }
        print(json.dumps({"type_id": type_id, "layer": fit.layer, "quality": fit.quality, "prefer": prefer}))
    (_OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    files = []
    for path in sorted(_OUT.glob("*.npz")):
        files.append({"path": str(path.relative_to(_ROOT)), "sha256": _sha256(path)})
    manifest = _ROOT / "configs" / "manifests" / "fly.yaml"
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw.setdefault("tiers", {})
    raw["tiers"]["prefer"] = {
        "description": "Optional efficiency / neutral / fidelity fits. Not the published default.",
        "artifacts": files,
    }
    manifest.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return 0 if all(item["quality"] == "ok" for item in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
