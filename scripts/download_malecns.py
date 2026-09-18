#!/usr/bin/env python3
"""Download MaleCNS v1.0 Feather trio into data/malecns/ (RFC-001 D3)."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = (
    {
        "name": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "approx_bytes": 13_000_000,
        "required": True,
    },
    {
        "name": "body-neurotransmitters-male-cns-v1.0.feather",
        "approx_bytes": 42_000_000,
        "required": True,
    },
    {
        "name": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "approx_bytes": 1_100_000_000,
        "required": False,  # ~1.1 GB; skip with --skip-weights
    },
)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"GET {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        total = resp.headers.get("Content-Length")
        n_total = int(total) if total else None
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if n_total:
                pct = 100.0 * done / n_total
                print(f"\r  {done / 1e6:.1f} / {n_total / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print()
    tmp.replace(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/malecns"),
        help="Destination directory (default: data/malecns)",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip the ~1.1 GB connectome-weights file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args(argv)

    for item in FILES:
        if not item["required"] and args.skip_weights:
            print(f"skip {item['name']}")
            continue
        dest = args.out / item["name"]
        if dest.exists() and not args.force:
            digest = _sha256(dest)
            print(f"ok  {dest}  sha256={digest[:16]}…  size={dest.stat().st_size}")
            continue
        url = f"{BASE}/{item['name']}"
        try:
            _download(url, dest)
        except Exception as exc:  # noqa: BLE001 — CLI surface
            print(f"FAILED {item['name']}: {exc}", file=sys.stderr)
            return 1
        digest = _sha256(dest)
        side = dest.with_suffix(dest.suffix + ".sha256")
        side.write_text(f"{digest}  {dest.name}\n", encoding="utf-8")
        size = dest.stat().st_size
        if size < 0.5 * float(item["approx_bytes"]):
            print(f"WARN {dest.name}: size {size} looks small (expected ~{item['approx_bytes']})")
        print(f"wrote {dest}  sha256={digest}")
    print("Done. See data/README.md for path conventions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
