#!/usr/bin/env python3
"""A1: build Colab notebook from jupytext-style markdown source (D21)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _ROOT / "examples" / "colab_src" / "demo.md"


def _md_to_notebook(md: str) -> dict:
    """Very small converter: split on '# %%' cell markers (py:percent-like)."""
    chunks = md.split("\n# %%")
    cells = []
    for i, chunk in enumerate(chunks):
        text = chunk if i == 0 else "# %%" + chunk
        text = text.strip("\n")
        if not text.strip():
            continue
        # First line may be `# %% [markdown]` or `# %%`
        lines = text.splitlines()
        cell_type = "markdown"
        if lines and lines[0].startswith("# %%"):
            header = lines[0]
            body = "\n".join(lines[1:]).strip("\n")
            if "markdown" in header:
                cell_type = "markdown"
            else:
                cell_type = "code"
        else:
            body = text
            cell_type = "markdown"
        source = body + "\n" if body and not body.endswith("\n") else body
        cells.append(
            {
                "cell_type": cell_type,
                "metadata": {},
                "source": [ln + "\n" for ln in source.splitlines()] or [""],
            }
        )
        if cell_type == "code":
            cells[-1]["outputs"] = []
            cells[-1]["execution_count"] = None
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=_DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=Path("assets/demo/demo.ipynb"))
    args = parser.parse_args(argv)
    if not args.src.exists():
        print(f"missing source: {args.src}")
        return 1
    nb = _md_to_notebook(args.src.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cells": len(nb["cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
