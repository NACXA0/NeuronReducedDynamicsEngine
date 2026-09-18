"""A-layer presets: nrde.make / nrde.demo (RFC-002 §3.2)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nrde.fitting.artifacts import resolve_artifact_path
from nrde.fitting.fit import fit_fi, load_fit
from nrde.io.connectome import erdos_renyi_graph
from nrde.types import FittedActivation, GraphData

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DEMO_CFG = _ROOT / "configs" / "flygym_demo.yaml"


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _ROOT / p


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_seed_fit(
    path: Path,
    expected_hash: str | None = None,
    *,
    type_id: str | None = None,
    model: str | None = None,
) -> FittedActivation:
    resolved = path
    if not resolved.exists() and (type_id or model):
        found = resolve_artifact_path(resolved.parent if resolved.suffix else resolved, type_id=type_id, model=model)
        if found is not None:
            resolved = found
    if resolved.exists():
        fit = load_fit(resolved)
        if expected_hash and fit.config_hash and fit.config_hash != expected_hash:
            warnings.warn(
                f"artifact hash mismatch for {resolved.name}: "
                f"got {fit.config_hash}, expected {expected_hash}",
                stacklevel=2,
            )
        return fit
    # Fall back: synthesize a cheap L0 table so demo works without committed seeds.
    use_model = model or "explif"
    stem = resolved.stem.lower()
    if use_model == "explif":
        if "adexp" in stem or stem == "acc":
            use_model = "adexp"
        elif "hh" in stem:
            use_model = "hh"
        elif stem in {"lif", "type_1"} or "inhibit" in stem:
            use_model = "lif"
    return fit_fi(
        use_model,
        type_id=type_id or resolved.stem,
        I_min=0.0,
        I_max=0.8 if use_model != "hh" else 1.6,
        n_I=10,
        t_total=400.0,
        window=250.0,
        dt=0.05 if use_model != "hh" else 0.02,
        n_validate=3,
    )


def make(name: str, **kwargs: Any) -> Any:
    """Gym-style factory. Supported: ``flygym-demo-v01``."""
    if name in {"flygym-demo-v01", "flygym"}:
        return make_flygym_demo(**kwargs)
    raise KeyError(f"Unknown preset {name!r}. Known: flygym-demo-v01")


def demo(name: str = "flygym", steps: int | None = None, **kwargs: Any) -> dict[str, Any]:
    """One-shot A-layer entry used by ``nrde demo`` and ``nrde.demo``."""
    env = make(name if name.endswith("-v01") or name == "flygym-demo-v01" else f"{name}-demo-v01", **kwargs)
    n = int(steps if steps is not None else getattr(env, "default_steps", 100))
    return env.run(n)


def make_flygym_demo(
    config: str | Path | None = None,
    headless: bool = True,
    visual: str | None = None,
    steps: int | None = None,
    **_unused: Any,
) -> Any:
    """Build EmbodiedEnv closed-loop demo (mock FlyGym if extra missing)."""
    from nrde.adapters.flygym import MockFlyGymSim, NRDEFlyGymEnv, maybe_make_flygym_sim

    cfg_path = _repo_path(config) if config else _DEFAULT_DEMO_CFG
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    n_nodes = int(cfg.get("n_nodes", 40))
    n_act = int(cfg.get("n_actuators", 6))
    n_sense = int(cfg.get("n_sense", 8))
    seed = int(cfg.get("seed", 0))
    default_steps = int(steps if steps is not None else cfg.get("steps", 100))

    art = cfg.get("artifacts") or [{"path": "artifacts/type_0.npz", "type_id": "type_0", "model": "explif"}]
    hashes = cfg.get("artifact_hashes") or {}
    fits: list[FittedActivation] = []
    for item in art:
        if isinstance(item, str):
            rel, tid, model = item, None, None
        else:
            rel = str(item.get("path") or item.get("artifact"))
            tid = item.get("type_id")
            model = item.get("model")
        p = _repo_path(rel)
        key = Path(rel).name
        fits.append(
            _load_seed_fit(
                p,
                expected_hash=hashes.get(key) or hashes.get(Path(p).name),
                type_id=tid,
                model=model,
            )
        )
    # Duplicate last fit if graph has more types than tables.
    while len(fits) < 2:
        fits.append(
            fits[0]
            if fits
            else _load_seed_fit(
                _repo_path("artifacts/type_0.npz"),
                type_id="type_0",
                model="explif",
            )
        )

    g: GraphData = erdos_renyi_graph(
        n_nodes,
        p=float(cfg.get("edge_p", 0.05)),
        n_types=2,
        seed=seed,
        weight=float(cfg.get("edge_weight", 0.001)),
        type_names=tuple(cfg.get("type_names") or ("type_0", "type_1")),
    )

    sense_idx = np.asarray(cfg.get("sense_idx") or list(range(n_sense)), dtype=int)
    motor_idx = np.asarray(
        cfg.get("motor_idx") or list(range(n_sense, n_sense + n_act)),
        dtype=int,
    )

    Factory = maybe_make_flygym_sim()
    if headless or Factory is MockFlyGymSim:
        sim: Any = MockFlyGymSim(n_actuators=n_act)
    else:
        try:
            sim = Factory(n_actuators=n_act)
        except TypeError:
            sim = MockFlyGymSim(n_actuators=n_act)

    if visual:
        # Soft stimulus cue for mock vision (A-layer only; no behavior claim).
        level = 0.9 if "on" in str(visual).lower() else 0.2
        if hasattr(sim, "_vision"):
            sim._vision = np.ones_like(sim._vision) * level

    env = NRDEFlyGymEnv(
        sim,
        g,
        fits[: len(g.type_names)],
        sense_idx=sense_idx,
        motor_idx=motor_idx,
        n_actuators=n_act,
        mode=str(cfg.get("mode", "rate")),
        fly_name=str(cfg.get("fly_name", "fly")),
        actuator_type=str(cfg.get("actuator_type", "joints")),
    )
    env.default_steps = default_steps  # type: ignore[attr-defined]
    return env
