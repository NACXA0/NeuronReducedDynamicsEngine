#!/usr/bin/env python3
"""README / M1 evidence figures from the published fly default dataset.

Recomputes f-I, chirp, and wall-clock against `datasets/fly/`. Does not rewrite
those tables. Ledger: `docs/reports/figures/evidence.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from nrde.binding import tables_for_graph
from nrde.engine.rate import run_rate
from nrde.fetch import fetch_preset
from nrde.fitting.chirp import chirp_impedance
from nrde.fitting.fit import load_fit
from nrde.io.connectome import load_connectome
from nrde.models.lif import analytic_rate
from nrde.sim import firing_rate, simulate_spikes
from nrde.validation import r2_score

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "docs" / "reports" / "figures"
_ACC = _ROOT / "datasets" / "fly" / "fits" / "aCC_l0.npz"
_LIF = _ROOT / "datasets" / "fly" / "fits" / "default_inhibitory_l0.npz"
_EXC = _ROOT / "datasets" / "fly" / "fits" / "default_excitatory_l0.npz"
_GRAPH = _ROOT / "datasets" / "fly" / "sparse" / "accessory_mn_circuit.feather"
_ANN = _ROOT / "datasets" / "fly" / "sparse" / "accessory_mn_annotations.feather"

# Match scripts/export_fly_default_dataset.py::_fit_model
_FI_T_TOTAL = 800.0
_FI_WINDOW = 400.0
_FI_DT = 0.05
_TIMING_T_MS = 200.0
_TIMING_STEPS = 200
_TIMING_ODE_CELLS = 8
_I_DRIVE = 0.40

_ODE = "#1f4e79"
_F = "#c45c26"
_LIF_C = "#6b7280"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or "unknown"
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _save_csv(path: Path, header: str, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def _median_ms(fn, repeats: int) -> float:
    samples: list[float] = []
    fn()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _map(value: float, lo: float, hi: float, a: float, b: float) -> float:
    if hi <= lo:
        return 0.5 * (a + b)
    t = (value - lo) / (hi - lo)
    return a + t * (b - a)


def _log_map(value: float, lo: float, hi: float, a: float, b: float) -> float:
    return _map(math.log10(max(value, lo)), math.log10(lo), math.log10(hi), a, b)


def write_svg(path: Path, body: str, width: int, height: int) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n"
        f"{body}\n</svg>\n",
        encoding="utf-8",
    )


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / max(n - 1, 1)
    return [lo + i * step for i in range(n)]


def svg_fi_overlay(path: Path, I: np.ndarray, r_ode: np.ndarray, r_f: np.ndarray, r_lif: np.ndarray) -> None:
    w, h = 720, 420
    l, r, t, b = 72.0, 24.0, 36.0, 56.0
    x0, x1, y0, y1 = l, w - r, h - b, t
    xmax = float(np.max(I))
    ymax = float(max(np.max(r_ode), np.max(r_f), np.max(r_lif), 1.0)) * 1.08

    def X(v: float) -> float:
        return _map(v, 0.0, xmax, x0, x1)

    def Y(v: float) -> float:
        return _map(v, 0.0, ymax, y0, y1)

    parts = [
        f'<text x="{w/2:.1f}" y="24" text-anchor="middle" font-family="sans-serif" '
        f'font-size="16" fill="#111827">AdExp f–I: online ODE vs NRDE F vs unified LIF</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111827" stroke-width="1.2"/>',
        f'<text x="{(x0+x1)/2:.1f}" y="{h-14}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#111827">I (nA)</text>',
        f'<text x="18" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#111827" transform="rotate(-90 18 {(y0+y1)/2:.1f})">rate (Hz)</text>',
    ]
    for xv in _ticks(0.0, xmax):
        parts.append(
            f'<line x1="{X(xv):.2f}" y1="{y0}" x2="{X(xv):.2f}" y2="{y0+5}" stroke="#111827"/>'
            f'<text x="{X(xv):.2f}" y="{y0+20}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11" fill="#374151">{xv:.2f}</text>'
        )
    for yv in _ticks(0.0, ymax):
        parts.append(
            f'<line x1="{x0-5}" y1="{Y(yv):.2f}" x2="{x0}" y2="{Y(yv):.2f}" stroke="#111827"/>'
            f'<text x="{x0-10}" y="{Y(yv)+4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11" fill="#374151">{yv:.0f}</text>'
        )

    def poly(xs: np.ndarray, ys: np.ndarray, color: str, dash: str | None = None) -> str:
        pts = " ".join(f"{X(float(a)):.2f},{Y(float(b)):.2f}" for a, b in zip(xs, ys))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{extra} points="{pts}"/>'

    parts.append(poly(I, r_ode, _ODE))
    parts.append(poly(I, r_f, _F, "7 5"))
    parts.append(poly(I, r_lif, _LIF_C, "2 5"))
    legend = [
        (_ODE, "solid", "AdExp ODE (ground truth)"),
        (_F, "7 5", "NRDE F (aCC table)"),
        (_LIF_C, "2 5", "unified LIF"),
    ]
    lx, ly = x0 + 12, y1 + 8
    for i, (color, dash, label) in enumerate(legend):
        yy = ly + i * 18
        extra = f' stroke-dasharray="{dash}"' if dash != "solid" else ""
        parts.append(
            f'<line x1="{lx}" y1="{yy}" x2="{lx+28}" y2="{yy}" stroke="{color}" '
            f'stroke-width="2.4"{extra}/>'
            f'<text x="{lx+34}" y="{yy+4}" font-family="sans-serif" font-size="12" '
            f'fill="#111827">{escape(label)}</text>'
        )
    write_svg(path, "\n".join(parts), w, h)


def svg_chirp(path: Path, freq: np.ndarray, z_adexp: np.ndarray, z_lif: np.ndarray) -> None:
    w, h = 720, 360
    l, r, t, b = 72.0, 24.0, 36.0, 56.0
    x0, x1, y0, y1 = l, w - r, h - b, t
    fmin, fmax = float(np.min(freq)), float(np.max(freq))
    zmax = float(max(np.max(z_adexp), np.max(z_lif), 1.0)) * 1.12

    def X(v: float) -> float:
        return _log_map(v, max(fmin, 0.5), fmax, x0, x1)

    def Y(v: float) -> float:
        return _map(v, 0.0, zmax, y0, y1)

    parts = [
        f'<text x="{w/2:.1f}" y="24" text-anchor="middle" font-family="sans-serif" '
        f'font-size="16" fill="#111827">Subthreshold |Z(ω)| (AdExp vs LIF)</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111827" stroke-width="1.2"/>',
        f'<text x="{(x0+x1)/2:.1f}" y="{h-14}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#111827">frequency (Hz, log)</text>',
        f'<text x="18" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" fill="#111827" transform="rotate(-90 18 {(y0+y1)/2:.1f})">|Z| (mV/nA)</text>',
    ]
    for xv in list(freq):
        parts.append(
            f'<text x="{X(float(xv)):.2f}" y="{y0+18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11" fill="#374151">{float(xv):.0f}</text>'
        )
    for yv in _ticks(0.0, zmax, 4):
        parts.append(
            f'<text x="{x0-8}" y="{Y(yv)+4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11" fill="#374151">{yv:.0f}</text>'
        )

    def poly(xs: np.ndarray, ys: np.ndarray, color: str, dash: str | None = None) -> str:
        pts = " ".join(f"{X(float(a)):.2f},{Y(float(b)):.2f}" for a, b in zip(xs, ys))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{extra} points="{pts}"/>'

    parts.append(poly(freq, z_adexp, _ODE))
    parts.append(poly(freq, z_lif, _LIF_C, "2 5"))
    parts.append(
        f'<line x1="{x0+12}" y1="{y1+12}" x2="{x0+40}" y2="{y1+12}" stroke="{_ODE}" stroke-width="2.4"/>'
        f'<text x="{x0+46}" y="{y1+16}" font-family="sans-serif" font-size="12">AdExp</text>'
        f'<line x1="{x0+120}" y1="{y1+12}" x2="{x0+148}" y2="{y1+12}" stroke="{_LIF_C}" '
        f'stroke-width="2.4" stroke-dasharray="2 5"/>'
        f'<text x="{x0+154}" y="{y1+16}" font-family="sans-serif" font-size="12">LIF</text>'
    )
    write_svg(path, "\n".join(parts), w, h)


def svg_pareto(path: Path, points: list[tuple[str, float, float, str]]) -> None:
    """points: (label, wall_ms, rmse_Hz, color). ODE uses error=0 plotted as floor."""
    w, h = 720, 420
    l, r, t, b = 84.0, 28.0, 40.0, 64.0
    x0, x1, y0, y1 = l, w - r, h - b, t
    walls = [p[1] for p in points]
    errs = [max(p[2], 1e-5) for p in points]
    xmin, xmax = min(walls) / 3.0, max(walls) * 3.0
    ymin, ymax = min(errs) / 3.0, max(errs) * 3.0
    xmin = max(xmin, 1e-3)
    ymin = max(ymin, 1e-5)

    def X(v: float) -> float:
        return _log_map(v, xmin, xmax, x0, x1)

    def Y(v: float) -> float:
        return _log_map(max(v, ymin), ymin, ymax, y0, y1)

    parts = [
        f'<text x="{w/2:.1f}" y="26" text-anchor="middle" font-family="sans-serif" '
        f'font-size="16" fill="#111827">Accuracy vs cost (AdExp cell, MaleCNS 249-node circuit)</text>',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111827" stroke-width="1.2"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#111827" stroke-width="1.2"/>',
        f'<text x="{(x0+x1)/2:.1f}" y="{h-16}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13">wall-clock for 249 cells × 200 ms (ms, log)</text>',
        f'<text x="18" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13" transform="rotate(-90 18 {(y0+y1)/2:.1f})">RMSE vs AdExp ODE (Hz, log)</text>',
    ]
    for exp in range(int(math.floor(math.log10(xmin))), int(math.ceil(math.log10(xmax))) + 1):
        xv = 10.0 ** exp
        if xmin <= xv <= xmax:
            parts.append(
                f'<line x1="{X(xv):.2f}" y1="{y0}" x2="{X(xv):.2f}" y2="{y0+5}" stroke="#111827"/>'
                f'<text x="{X(xv):.2f}" y="{y0+20}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="11" fill="#374151">{xv:g}</text>'
            )
    for exp in range(int(math.floor(math.log10(ymin))), int(math.ceil(math.log10(ymax))) + 1):
        yv = 10.0 ** exp
        if ymin <= yv <= ymax:
            label = "0" if yv <= 1.5e-5 else f"{yv:g}"
            parts.append(
                f'<text x="{x0-8}" y="{Y(yv)+4:.2f}" text-anchor="end" font-family="sans-serif" '
                f'font-size="11" fill="#374151">{label}</text>'
            )
    for label, wall, err, color in points:
        plot_err = max(err, 1e-5)
        cx, cy = X(wall), Y(plot_err)
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="{color}"/>')
        parts.append(
            f'<text x="{cx+10:.2f}" y="{cy-8:.2f}" font-family="sans-serif" font-size="12" '
            f'fill="#111827">{escape(label)}</text>'
        )
    write_svg(path, "\n".join(parts), w, h)


def _tables_for_circuit(graph, acc, exc, inh, uniform_lif: bool):
    if uniform_lif:
        by_name = {name: inh for name in graph.type_names}
        return tables_for_graph(graph, by_name, default_key=inh.type_id)
    by_name = {}
    for name in graph.type_names:
        by_name[name] = acc if "flexor MN" in name else exc
    return tables_for_graph(graph, by_name, default_key=exc.type_id)


def compute(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetch_preset("fly", tier="pretrained")
    if not fetch.get("ok"):
        raise SystemExit(f"pretrained fetch failed: {json.dumps(fetch)[:800]}")

    acc = load_fit(_ACC)
    lif_fit = load_fit(_LIF)
    exc = load_fit(_EXC)
    I_train = np.asarray(acc.I_grid, dtype=np.float64)
    I_eval = 0.5 * (I_train[:-1] + I_train[1:])

    r_ode = np.array(
        [
            firing_rate("adexp", acc.params or None, float(i), t_total=_FI_T_TOTAL, window=_FI_WINDOW, dt=_FI_DT)
            for i in I_eval
        ],
        dtype=np.float64,
    )
    r_f, _ = acc.eval_rate(I_eval)
    r_lif_num = np.array(
        [
            firing_rate("lif", None, float(i), t_total=_FI_T_TOTAL, window=_FI_WINDOW, dt=_FI_DT)
            for i in I_eval
        ],
        dtype=np.float64,
    )
    r_lif_analytic = np.array([analytic_rate(float(i)) for i in I_eval], dtype=np.float64)
    r_f_train, _ = acc.eval_rate(I_train)

    r2_f = float(r2_score(r_ode, r_f))
    r2_lif = float(r2_score(r_ode, r_lif_num))
    mse_f = float(np.mean((r_ode - r_f) ** 2))
    mse_lif = float(np.mean((r_ode - r_lif_num) ** 2))
    rmse_f = float(math.sqrt(mse_f))
    rmse_lif = float(math.sqrt(mse_lif))

    stored_z = None
    if acc.z_freqs is not None and acc.z_abs is not None:
        stored_z = {"freq_Hz": acc.z_freqs.tolist(), "Z_abs": acc.z_abs.tolist()}
    z_adexp = chirp_impedance("adexp", acc.params or None)
    z_lif = chirp_impedance("lif", None)

    graph = load_connectome(_GRAPH, annotations_path=_ANN)
    I_ext = np.full(graph.n_nodes, _I_DRIVE, dtype=np.float64)
    tables_f = _tables_for_circuit(graph, acc, exc, lif_fit, uniform_lif=False)
    tables_lif = _tables_for_circuit(graph, acc, exc, lif_fit, uniform_lif=True)

    def _run(tables):
        run_rate(graph, tables, n_steps=_TIMING_STEPS, I_ext=I_ext, record_trace=False)

    wall_f = _median_ms(lambda: _run(tables_f), repeats=3)
    wall_lif = _median_ms(lambda: _run(tables_lif), repeats=3)
    state_f, _ = run_rate(graph, tables_f, n_steps=_TIMING_STEPS, I_ext=I_ext, record_trace=False)
    state_lif, _ = run_rate(graph, tables_lif, n_steps=_TIMING_STEPS, I_ext=I_ext, record_trace=False)

    simulate_spikes("adexp", acc.params or None, _I_DRIVE, t_total=_TIMING_T_MS, dt=_FI_DT)

    def _ode_probe():
        for _ in range(_TIMING_ODE_CELLS):
            simulate_spikes("adexp", acc.params or None, _I_DRIVE, t_total=_TIMING_T_MS, dt=_FI_DT)

    wall_ode_probe = _median_ms(_ode_probe, repeats=3)
    wall_ode = wall_ode_probe * (graph.n_nodes / _TIMING_ODE_CELLS)

    fi_csv = out_dir / "adexp_fi_overlay.csv"
    chirp_csv = out_dir / "adexp_lif_chirp.csv"
    pareto_csv = out_dir / "hero_pareto.csv"
    _save_csv(
        fi_csv,
        "I_nA,r_ode_Hz,r_F_Hz,r_lif_Hz,r_lif_analytic_Hz",
        np.column_stack([I_eval, r_ode, r_f, r_lif_num, r_lif_analytic]),
    )
    _save_csv(
        chirp_csv,
        "freq_Hz,Z_adexp,Z_lif",
        np.column_stack([z_adexp.freqs_hz, z_adexp.z_abs, z_lif.z_abs]),
    )
    pareto_csv.write_text(
        "method,wall_ms,rmse_Hz,r2_vs_adexp_ode\n"
        f"ode_adexp,{wall_ode:.6f},0,1\n"
        f"nrde_F,{wall_f:.6f},{rmse_f:.8f},{r2_f:.8f}\n"
        f"unified_lif,{wall_lif:.6f},{rmse_lif:.8f},{r2_lif:.8f}\n",
        encoding="utf-8",
    )

    fi_svg = out_dir / "adexp_fi_overlay.svg"
    chirp_svg = out_dir / "adexp_lif_chirp.svg"
    pareto_svg = out_dir / "hero_pareto.svg"
    svg_fi_overlay(fi_svg, I_eval, r_ode, r_f, r_lif_num)
    svg_chirp(chirp_svg, z_adexp.freqs_hz, z_adexp.z_abs, z_lif.z_abs)
    svg_pareto(
        pareto_svg,
        [
            ("online AdExp ODE", wall_ode, 0.0, _ODE),
            ("NRDE F", wall_f, rmse_f, _F),
            ("unified LIF", wall_lif, rmse_lif, _LIF_C),
        ],
    )

    dataset_files = []
    for item in fetch.get("artifacts") or []:
        rel = str(item.get("path") or "")
        if not rel:
            continue
        dataset_files.append(
            {
                "path": rel,
                "sha256": item.get("sha256") or _sha256(_ROOT / rel),
                "status": item.get("status"),
            }
        )
    outputs = [
        fi_csv,
        chirp_csv,
        pareto_csv,
        fi_svg,
        chirp_svg,
        pareto_svg,
    ]
    ledger = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": "uv run --no-sync python scripts/make_figures.py",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": _cpu_model(),
            "nrde_engine": os.environ.get("NRDE_ENGINE", "numpy"),
        },
        "dataset": {
            "tier": "pretrained",
            "fetch_ok": True,
            "n_artifacts": len(dataset_files),
            "ground_truth_fit": "datasets/fly/fits/aCC_l0.npz",
            "lif_fit": "datasets/fly/fits/default_inhibitory_l0.npz",
            "circuit": "datasets/fly/sparse/accessory_mn_circuit.feather",
            "files": dataset_files,
        },
        "protocol": {
            "ground_truth": "AdExp Euler, dt=0.05 ms; error relative to this ODE",
            "fi": {
                "t_total_ms": _FI_T_TOTAL,
                "window_ms": _FI_WINDOW,
                "dt_ms": _FI_DT,
                "I_eval": "midpoints of aCC_l0 I_grid (held-out vs training knots)",
                "n_eval": int(I_eval.size),
            },
            "timing": {
                "n_nodes": int(graph.n_nodes),
                "n_edges": int(graph.edge_weight.size),
                "n_steps": _TIMING_STEPS,
                "engine_dt_ms": 1.0,
                "simulated_ms": _TIMING_T_MS,
                "ode_dt_ms": _FI_DT,
                "ode_cells_measured": _TIMING_ODE_CELLS,
                "ode_wall_scaled_to_n_nodes": True,
                "I_ext_nA": _I_DRIVE,
                "repeats": 3,
                "stat": "median",
            },
            "notes": [
                "aCC_l0.npz stores the AdExp steady f-I; fitter labels layer L1 because of adaptation.",
                "T4 tables clone aCC; not a second electrophysiology source.",
                "Circuit X-axis is typed F vs uniform LIF lookup on the 249-node accessory MN graph.",
                "ODE wall-clock is 8 cells × 200 ms Euler, linearly scaled to 249 cells.",
            ],
        },
        "results": {
            "r2_F_vs_ode": r2_f,
            "r2_lif_vs_ode": r2_lif,
            "r2_F_meta": float(acc.r2),
            "mse_F": mse_f,
            "mse_lif": mse_lif,
            "rmse_F_Hz": rmse_f,
            "rmse_lif_Hz": rmse_lif,
            "nfr4_pass": bool(r2_f >= 0.98),
            "acc_layer_in_meta": acc.layer,
            "wall_ms": {
                "ode_adexp_probe": wall_ode_probe,
                "ode_adexp_scaled": wall_ode,
                "nrde_F_circuit": wall_f,
                "lif_F_circuit": wall_lif,
            },
            "speedup_vs_ode": {
                "nrde_F": wall_ode / wall_f if wall_f > 0 else None,
                "lif_F": wall_ode / wall_lif if wall_lif > 0 else None,
            },
            "circuit_mean_rate_Hz": {
                "typed_F": float(np.mean(state_f.r)),
                "uniform_lif": float(np.mean(state_lif.r)),
            },
            "chirp": {
                "adexp_has_peak": bool(z_adexp.has_peak),
                "adexp_peak_hz": z_adexp.peak_hz,
                "adexp_notes": z_adexp.notes,
                "lif_has_peak": bool(z_lif.has_peak),
                "lif_notes": z_lif.notes,
                "stored_in_aCC_npz": stored_z,
            },
            "train_knots_r_F_first3": r_f_train[:3].tolist(),
        },
        "outputs": [{"path": str(p.relative_to(_ROOT)), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in outputs],
    }
    ledger_path = out_dir / "evidence.json"
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def verify(out_dir: Path) -> int:
    ledger_path = out_dir / "evidence.json"
    if not ledger_path.exists():
        print("missing docs/reports/figures/evidence.json")
        return 1
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    failed: list[str] = []
    fetch = fetch_preset("fly", tier="pretrained")
    if not fetch.get("ok"):
        failed.append("pretrained fetch not ok")
    n = int((ledger.get("dataset") or {}).get("n_artifacts") or 0)
    if n != 33:
        failed.append(f"expected 33 pretrained artifacts in ledger, got {n}")
    for item in (ledger.get("dataset") or {}).get("files") or []:
        path = _ROOT / str(item["path"])
        if not path.exists():
            failed.append(f"missing {item['path']}")
            continue
        digest = _sha256(path)
        if item.get("sha256") and digest != item["sha256"]:
            failed.append(f"hash mismatch {item['path']}")
    for item in ledger.get("outputs") or []:
        path = _ROOT / str(item["path"])
        if not path.exists():
            failed.append(f"missing output {item['path']}")
            continue
        if _sha256(path) != item["sha256"]:
            failed.append(f"output hash mismatch {item['path']}")
    results = ledger.get("results") or {}
    if not results.get("nfr4_pass"):
        failed.append("ledger nfr4_pass is false")
    if failed:
        print("\n".join(failed))
        return 1
    print("evidence ledger ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument("--verify", action="store_true", help="Check hashes; do not recompute")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else _ROOT / args.out
    if args.verify:
        return verify(out)
    ledger = compute(out)
    print(
        json.dumps(
            {
                "r2_F_vs_ode": ledger["results"]["r2_F_vs_ode"],
                "r2_lif_vs_ode": ledger["results"]["r2_lif_vs_ode"],
                "wall_ms": ledger["results"]["wall_ms"],
                "speedup_vs_ode": ledger["results"]["speedup_vs_ode"],
                "n_artifacts": ledger["dataset"]["n_artifacts"],
                "out": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
