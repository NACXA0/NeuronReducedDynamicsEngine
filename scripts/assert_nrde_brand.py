#!/usr/bin/env python3
"""Fail if legacy product-brand tokens appear in the tree.

NRDE brand only. Method term "state approximation" is allowed.
Functional-requirement IDs (FR-1, …) are allowed — they are not a product brand.

Forbidden tokens are stored encoded so this repo does not keep readable legacy
product appellations in source.

Usage:
  python scripts/assert_nrde_brand.py
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".tox",
    "dist",
    "build",
}
SKIP_SUFFIXES = {
    ".npz",
    ".npy",
    ".png",
    ".gif",
    ".jpg",
    ".jpeg",
    ".webp",
    ".pdf",
    ".whl",
    ".so",
    ".pyc",
    ".pyo",
    ".feather",
    ".parquet",
    ".zip",
    ".tar",
    ".gz",
}
SKIP_RELPATHS = {
    "scripts/assert_nrde_brand.py",
    "tests/test_brand_naming.py",
}


def _b(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8")


# base64 of former product tokens (do not store plaintext appellations here)
_FORBIDDEN_B64: list[tuple[str, bool]] = [
    ("ZnJlLW5ldXJvbi1lbmdpbmU=", False),
    ("Zmx5LXJlZHVjZWQtZW5naW5l", False),
    ("ZnJlX25ldXJvbl9lbmdpbmU=", False),
    ("Zmx5X3JlZHVjZWRfZW5naW5l", False),
    ("U3RhdGVBcHByb3hOZXVy", False),
    ("RlJFRmx5R3ltRW52", False),
    ("ZnJlX3ZlcnNpb24=", False),
    ("ZnJlLWRlbW8tYXNzZXRz", False),
    ("ZnJlLWZseS1vZmZsaW5l", False),
    ("5p6c6J2H6ZmN57u05Yqo5Yqb5a2m5byV5pOO", False),
    ("Rmx5IFJlZHVjZWQtZHluYW1pY3MgRW5naW5l", False),
    ("Rmx5IFJlZHVjZWQtRHluYW1pY3MgRW5naW5l", False),
    ("XGJGUkVcYg==", True),
    ("XGJmcmVcYg==", True),
]


def iter_text_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(root).parts
        if any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in parts[:-1]):
            continue
        rel = str(p.relative_to(root))
        if rel in SKIP_RELPATHS:
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield p, rel


def scan() -> list[str]:
    hits: list[str] = []
    rules = [(_b(raw), is_re) for raw, is_re in _FORBIDDEN_B64]
    for path, rel in iter_text_files(ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern, is_re in rules:
            if is_re:
                for m in re.finditer(pattern, text):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{rel}:{line}: legacy brand token")
            else:
                if pattern not in text:
                    continue
                idx = text.index(pattern)
                line = text.count("\n", 0, idx) + 1
                hits.append(f"{rel}:{line}: legacy brand token")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    hits = scan()
    if hits:
        print("legacy brand tokens found:", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        return 1
    print("ok: no legacy brand tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
