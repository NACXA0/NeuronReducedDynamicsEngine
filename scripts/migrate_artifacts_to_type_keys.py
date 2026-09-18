#!/usr/bin/env python3
"""Migrate artifacts/ from model-keyed stems to type-keyed stems (RFC-002 P2).

Examples:
  python scripts/migrate_artifacts_to_type_keys.py --dry-run
  python scripts/migrate_artifacts_to_type_keys.py --keep-legacy
  python scripts/migrate_artifacts_to_type_keys.py --no-keep-legacy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nrde.fitting.artifacts import migrate_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing *.npz (+ *.meta.json)",
    )
    parser.add_argument(
        "--include-layer",
        action="store_true",
        help="Append _{layer} to the type-keyed stem (default: type_id only)",
    )
    parser.add_argument(
        "--keep-legacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy to type key and keep model-keyed files (default: true)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite differing type-keyed targets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable report")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"missing directory: {args.root}", file=sys.stderr)
        return 1

    reports = migrate_directory(
        args.root,
        include_layer=args.include_layer,
        dry_run=args.dry_run,
        keep_legacy=args.keep_legacy,
        force=args.force,
    )
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for row in reports:
            print(f"{row['action']:18} {row['src']} → {row['dst']}  (type_id={row.get('type_id')})")
        n_mig = sum(1 for r in reports if r["action"] in {"copied", "renamed", "would_migrate"})
        print(f"done: {n_mig} migrate action(s), {len(reports)} file(s) scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
