#!/usr/bin/env python3
"""A1: assemble offline zip (seeds + manifest + config) for Release (D21)."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

SEED_FILES = (
    "artifacts/type_0.npz",
    "artifacts/type_0.meta.json",
    "artifacts/aCC.npz",
    "artifacts/aCC.meta.json",
    "artifacts/hh_demo.npz",
    "artifacts/hh_demo.meta.json",
    "configs/flygym_demo.yaml",
    "configs/manifests/fly.yaml",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dist/nrde-fly-offline.zip"))
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    manifest = {"files": [], "missing": []}
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in SEED_FILES:
            path = _ROOT / rel
            if not path.exists():
                manifest["missing"].append(rel)
                continue
            zf.write(path, arcname=rel)
            manifest["files"].append({"path": rel, "sha256": _sha256(path), "bytes": path.stat().st_size})
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps({"out": str(args.out), "n_files": len(manifest["files"]), "missing": manifest["missing"]}))
    return 0 if not manifest["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
