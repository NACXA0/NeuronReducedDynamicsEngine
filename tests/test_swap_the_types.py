"""P1: swap-the-types example keeps the fly kernel with different labels."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_swap_the_types_example(capsys):
    spec_path = ROOT / "examples" / "06_swap_the_types.py"
    spec = importlib.util.spec_from_file_location("swap_the_types", spec_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.main()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["bound"] == {"prey": "prey", "predator": "predator"}
    assert printed["bound"] == payload["bound"]
    assert payload["n_nodes"] == 40
    assert payload["mean_rate"] >= 0.0
