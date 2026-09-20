#!/usr/bin/env python3
"""Write the MaleCNS fly instance as the published default dataset.

Seeds under artifacts/ are not touched. L3/FNO is recorded as deferred (v0.2),
not invented. Full Feather files stay in data/malecns; this script keeps a
small labelled circuit plus every L0/L1/L2 table the fitter actually emits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import yaml

from nrde.calibration import (
    apply_alpha_scale,
    apply_shiu_weights,
    calibrate_alpha,
    relative_rate_error,
)
from nrde.cli import main as cli_main
from nrde.engine.rate import run_rate
from nrde.fitting.fit import attach_lut, fit_fi, fit_spike_lut, save_fit
from nrde.fitting.srm import attach_srm, fit_srm_kernels
from nrde.io.connectome import erdos_renyi_graph, load_connectome
from nrde.validation import validate_single

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "datasets" / "fly"
_MALE = _ROOT / "data" / "malecns"
_MOTOR_TYPES = ("Acc. ti flexor MN", "Acc. tr flexor MN")
_MODELS = (
    ("adexp", "aCC", 0.0, 0.8),
    ("explif", "default excitatory", 0.0, 0.8),
    ("lif", "default inhibitory", 0.0, 0.8),
)
_CLONE = ("aCC", "T4")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _stem(type_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in type_id)


def _extract_circuit() -> dict[str, int]:
    ann_path = _MALE / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    syn_path = _MALE / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    ann = feather.read_table(ann_path, columns=["bodyId", "type"])
    body = ann.column("bodyId").to_numpy()
    types = np.asarray(ann.column("type").to_pylist(), dtype=object)
    motor = body[np.isin(types, list(_MOTOR_TYPES))]
    motor_set = np.unique(motor.astype(np.int64))
    syn = feather.read_table(syn_path, columns=["body_pre", "body_post", "weight"])
    pre = syn.column("body_pre").to_numpy()
    post = syn.column("body_post").to_numpy()
    weight = syn.column("weight").to_numpy()
    del syn
    onto_motor = np.isin(post, motor_set)
    pre_m = pre[onto_motor]
    w_m = weight[onto_motor].astype(np.float64, copy=False)
    partners, inverse = np.unique(pre_m, return_inverse=True)
    strength = np.zeros(partners.shape[0], dtype=np.float64)
    np.add.at(strength, inverse, w_m)
    keep_partner = partners[~np.isin(partners, motor_set)]
    strength = strength[~np.isin(partners, motor_set)]
    order = np.argsort(strength)[::-1][:180]
    nodes = np.unique(np.concatenate([motor_set, keep_partner[order]]))
    both = np.isin(pre, nodes) & np.isin(post, nodes)
    sparse = _OUT / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    feather.write_feather(
        pa.table(
            {
                "body_pre": pre[both],
                "body_post": post[both],
                "weight": weight[both],
            }
        ),
        sparse / "accessory_mn_circuit.feather",
    )
    ann_keep = np.isin(body, nodes)
    feather.write_feather(
        pa.table({"bodyId": body[ann_keep], "type": types[ann_keep].astype(str)}),
        sparse / "accessory_mn_annotations.feather",
    )
    del pre, post, weight
    return {
        "n_motor": int(motor_set.size),
        "n_nodes": int(nodes.size),
        "n_edges": int(both.sum()),
    }


def _fit_model(model: str, type_id: str, i_min: float, i_max: float) -> dict[str, dict]:
    from nrde.fitting.chirp import chirp_impedance

    base = fit_fi(
        model,
        type_id=type_id,
        I_min=i_min,
        I_max=i_max,
        n_I=16,
        t_total=800.0,
        window=400.0,
        dt=0.05,
        n_validate=4,
        chirp=False,
    )
    z = chirp_impedance(model, None, I0=max(i_min, 0.02), amp=0.01) if model == "adexp" else None
    if z is not None:
        base.z_freqs = z.freqs_hz
        base.z_abs = z.z_abs
        base.notes = "; ".join(p for p in (base.notes, z.notes) if p)
    lut = fit_spike_lut(model, I_min=i_min, I_max=i_max, n_I=16)
    kernels = fit_srm_kernels(model, t_kernel=40.0, dt=0.1)
    variants = {
        "l0": replace(base, config_hash=None, git_commit=None),
        "l1": attach_lut(replace(base, config_hash=None, git_commit=None), lut),
        "l2": attach_srm(replace(base, config_hash=None, git_commit=None), kernels),
        "l1l2": attach_srm(attach_lut(replace(base, config_hash=None, git_commit=None), lut), kernels),
    }
    fits = _OUT / "fits"
    prec = _OUT / "precision"
    fits.mkdir(parents=True, exist_ok=True)
    prec.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    stem = _stem(type_id)
    for name, fit in variants.items():
        path = fits / f"{stem}_{name}.npz"
        save_fit(fit, path, skip_if_unchanged=False)
        np.savez_compressed(
            prec / f"{stem}_{name}_f32.npz",
            I_grid=np.asarray(fit.I_grid, dtype=np.float32),
            r_grid=np.asarray(fit.r_grid, dtype=np.float32),
            I_onset=np.asarray([fit.I_onset], dtype=np.float32),
        )
        report = validate_single(fit)
        reports[name] = {
            "path": str(path.relative_to(_ROOT)),
            "f32": str((prec / f"{stem}_{name}_f32.npz").relative_to(_ROOT)),
            "type_id": type_id,
            "model": model,
            "layer": fit.layer,
            "r2": fit.r2,
            "quality": fit.quality,
            "nfr4_pass": report.nfr4_pass,
            "relative_vp": report.relative_vp,
            "nfr5_pass": report.nfr5_pass,
            "config_hash": fit.config_hash,
            "dtypes": {"fits": "float64", "precision": "float32"},
            "chirp_bins": None if fit.z_freqs is None else int(fit.z_freqs.size),
        }
    return reports


def _clone_type(src: str, dst: str, reports: dict[str, dict]) -> dict[str, dict]:
    from nrde.fitting.fit import load_fit

    cloned: dict[str, dict] = {}
    for name, item in reports.items():
        fit = load_fit(_ROOT / item["path"])
        fit = replace(fit, type_id=dst, config_hash=None, git_commit=None, type_ids=(dst,))
        path = _OUT / "fits" / f"{_stem(dst)}_{name}.npz"
        save_fit(fit, path, skip_if_unchanged=False)
        np.savez_compressed(
            _OUT / "precision" / f"{_stem(dst)}_{name}_f32.npz",
            I_grid=np.asarray(fit.I_grid, dtype=np.float32),
            r_grid=np.asarray(fit.r_grid, dtype=np.float32),
            I_onset=np.asarray([fit.I_onset], dtype=np.float32),
        )
        cloned[name] = {**item, "path": str(path.relative_to(_ROOT)), "type_id": dst, "cloned_from": src}
        cloned[name]["f32"] = str((_OUT / "precision" / f"{_stem(dst)}_{name}_f32.npz").relative_to(_ROOT))
        cloned[name]["config_hash"] = fit.config_hash
    return cloned


def _alpha_report() -> dict[str, float | bool]:
    fit = fit_fi("lif", type_id="shiu_probe", I_min=0.0, I_max=0.8, n_I=16, t_total=300.0, window=180.0, n_validate=2)
    n = 32
    I_ext = np.full(n, 0.24)

    def circuit():
        g = erdos_renyi_graph(n, p=0.2, n_types=1, seed=2, weight=1.0, type_names=("default inhibitory",))
        apply_shiu_weights(g, synapse_counts=np.full(g.edge_weight.shape, 1.0))
        return g

    held = 3.0
    observed = circuit()
    apply_alpha_scale(observed, held)
    target, _ = run_rate(observed, [fit], n_steps=30, I_ext=I_ext, record_trace=False)
    start = circuit()
    scale, mse = calibrate_alpha(start, [fit], target.r, I_ext=I_ext, n_steps=30)
    apply_alpha_scale(start, scale)
    got, _ = run_rate(start, [fit], n_steps=30, I_ext=I_ext, record_trace=False)
    err = relative_rate_error(got.r, target.r)
    sparse = _OUT / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sparse / "shiu_labelled_circuit.npz",
        edge_index=start.edge_index,
        edge_weight_f32=np.asarray(start.edge_weight, dtype=np.float32),
        edge_weight_f64=np.asarray(start.edge_weight, dtype=np.float64),
        node_type=start.node_type,
        alpha_scale=np.asarray([scale]),
    )
    return {
        "start": "shiu_w_syn_mv_0.275",
        "held_out_scale": held,
        "recovered_scale": scale,
        "mse": mse,
        "relative_rate_error": err,
        "pass": bool(err <= 0.10),
    }


def _write_manifest(circuit: dict[str, int], fits: dict, alpha: dict, demo: dict) -> None:
    files = []
    for path in sorted(_OUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(_ROOT)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    male = []
    for name in (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "body-neurotransmitters-male-cns-v1.0.feather",
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    ):
        path = _MALE / name
        sidecar = Path(str(path) + ".sha256")
        digest = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.exists() else _sha256(path)
        male.append({"path": str(path.relative_to(_ROOT)), "sha256": digest, "bytes": path.stat().st_size})
    acceptance = {
        "gates": {
            "1_adexp_fi_chirp": {
                "pass": bool(fits["aCC"]["l0"]["nfr4_pass"] and fits["aCC"]["l0"]["chirp_bins"]),
                "r2": fits["aCC"]["l0"]["r2"],
                "chirp_bins": fits["aCC"]["l0"]["chirp_bins"],
            },
            "2_nfr5": {
                "pass": bool(fits["aCC"]["l1"]["nfr5_pass"] or fits["default inhibitory"]["l1"]["nfr5_pass"]),
                "aCC_l1_relative_vp": fits["aCC"]["l1"]["relative_vp"],
                "lif_l1_relative_vp": fits["default inhibitory"]["l1"]["relative_vp"],
            },
            "3_embodied": demo,
            "4_alpha": alpha,
            "5_coverage_pysr": {
            "pass": None,
            "note": "Filled by the test run, not by this export.",
        },
        "l3": {"pass": False, "status": "deferred_v0.2", "reason": "FNO / Volterra is not a v0.1 product"},
        },
        "fits": fits,
        "circuit": circuit,
        "full_sparse": male,
        "precision": {"activation_table": ["float64", "float32"], "edge_weight": ["float32", "float64"]},
    }
    reports = _OUT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    (reports / "l3.json").write_text(
        json.dumps(acceptance["gates"]["l3"], indent=2),
        encoding="utf-8",
    )
    files.append(
        {
            "path": "datasets/fly/reports/acceptance.json",
            "sha256": _sha256(reports / "acceptance.json"),
            "bytes": (reports / "acceptance.json").stat().st_size,
        }
    )
    files.append(
        {
            "path": "datasets/fly/reports/l3.json",
            "sha256": _sha256(reports / "l3.json"),
            "bytes": (reports / "l3.json").stat().st_size,
        }
    )
    manifest = {
        "preset": "fly",
        "role": "published_default",
        "instance": "MaleCNS v1.0 accessory flexor motoneurons",
        "files": files,
        "full_sparse": male,
    }
    (_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _patch_fetch_tier(files)


def _patch_fetch_tier(files: list[dict]) -> None:
    path = _ROOT / "configs" / "manifests" / "fly.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["tiers"]["pretrained"] = {
        "description": "Published MaleCNS default: L0-L2 tables, float32 copies, extracted circuit",
        "artifacts": [{"path": item["path"], "sha256": item["sha256"]} for item in files if item["path"].endswith(".npz")],
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    circuit = _extract_circuit()
    graph = load_connectome(
        _OUT / "sparse" / "accessory_mn_circuit.feather",
        annotations_path=_OUT / "sparse" / "accessory_mn_annotations.feather",
    )
    circuit["loaded_nodes"] = int(graph.n_nodes)
    circuit["loaded_edges"] = int(graph.edge_weight.size)
    fits: dict[str, dict] = {}
    for model, type_id, i_min, i_max in _MODELS:
        fits[type_id] = _fit_model(model, type_id, i_min, i_max)
    src, dst = _CLONE
    fits[dst] = _clone_type(src, dst, fits[src])
    alpha = _alpha_report()
    demo_rc = cli_main(["demo", "flygym", "--steps", "100", "--headless"])
    demo = {"steps": 100, "returncode": int(demo_rc), "backend": "mock_when_headless", "pass": demo_rc == 0}
    _write_manifest(circuit, fits, alpha, demo)
    nfr4 = fits["aCC"]["l0"]["nfr4_pass"]
    nfr5 = bool(fits["aCC"]["l1"]["nfr5_pass"] or fits["default inhibitory"]["l1"]["nfr5_pass"])
    print(json.dumps({"circuit": circuit, "alpha_pass": alpha["pass"], "nfr4": nfr4, "nfr5": nfr5, "demo": demo_rc}, indent=2))
    return 0 if alpha["pass"] and demo_rc == 0 and nfr4 and nfr5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
