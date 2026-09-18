#!/usr/bin/env python3
"""M1 figures: f-I overlay + chirp |Z(ω)| for AdExp / ExpLIF / HH (report assets)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fre.offline.chirp import chirp_impedance
from fre.offline.fit import fit_fi
from fre.sim import firing_rate


def _save_csv(path: Path, header: str, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def make_fi_overlay(model: str, out_dir: Path) -> Path:
    fit = fit_fi(
        model,
        type_id=model,
        I_min=0.0,
        I_max=0.9 if model != "hh" else 1.6,
        n_I=12,
        t_total=400.0,
        window=250.0,
        dt=0.05 if model != "hh" else 0.02,
        n_validate=4,
    )
    I = fit.I_grid
    r_ode = np.array([firing_rate(model, None, float(i), t_total=400.0, window=250.0, dt=0.05) for i in I])
    r_f, _ = fit.eval_rate(I)
    rows = np.column_stack([I, r_ode, r_f])
    path = out_dir / f"M1_{model}_fi.csv"
    _save_csv(path, "I_nA,r_ode_Hz,r_F_Hz", rows)
    return path


def make_chirp(model: str, out_dir: Path) -> Path:
    report = chirp_impedance(model, freqs_hz=np.array([1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]))
    rows = np.column_stack([report.freqs_hz, report.z_abs])
    path = out_dir / f"M1_{model}_chirp.csv"
    _save_csv(path, "freq_Hz,Z_abs", rows)
    note = out_dir / f"M1_{model}_chirp_note.txt"
    note.write_text(
        f"has_peak={report.has_peak}\npeak_hz={report.peak_hz}\nnotes={report.notes}\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/reports/figures"),
        help="Output directory for CSV / notes",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["adexp", "explif", "hh"],
        help="Models to plot",
    )
    args = parser.parse_args(argv)
    for model in args.models:
        fi = make_fi_overlay(model, args.out)
        z = make_chirp(model, args.out)
        print(f"{model}: {fi.name}, {z.name}")
    print("CSV written. Render with any plotting tool; see docs/reports/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
