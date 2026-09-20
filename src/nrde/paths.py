"""Locate the checkout or the directory where fetched data should land."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Directory that holds ``configs/manifests`` and, in a checkout, ``pyproject.toml``.

    Resolution order:

    1. ``NRDE_ROOT`` if set.
    2. Walk parents of this file until both ``pyproject.toml`` and ``configs/manifests`` exist
       (editable checkout: ``src/nrde/…``).
    3. The current working directory when it is a checkout.
    4. Otherwise the current working directory, so a pip install materializes data beside
       the command the user just ran.
    """
    env = os.environ.get("NRDE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs" / "manifests").is_dir():
            return parent
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "configs" / "manifests").is_dir():
        return cwd
    return cwd
