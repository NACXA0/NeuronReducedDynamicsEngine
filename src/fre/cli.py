"""CLI: fit, simulate, validate (IF-4)."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _cmd_fit(args: argparse.Namespace) -> int:
    from fre.offline.fit import attach_lut, fit_fi, fit_spike_lut, save_fit

    fit = fit_fi(
        args.model,
        type_id=args.type_id,
        I_min=args.I_min,
        I_max=args.I_max,
        n_I=args.n_I,
        t_total=args.t_total,
        window=args.window,
        dt=args.dt,
        chirp=args.chirp,
    )
    if args.lut:
        lut = fit_spike_lut(
            args.model,
            I_min=args.I_min,
            I_max=args.I_max,
            n_I=max(8, args.n_I // 2),
        )
        fit = attach_lut(fit, lut)
    if args.srm:
        from fre.offline.srm import attach_srm, fit_srm_kernels

        fit = attach_srm(fit, fit_srm_kernels(args.model, t_kernel=40.0, dt=0.1))
    path = save_fit(fit, args.out)
    print(json.dumps({"out": str(path), "r2": fit.r2, "quality": fit.quality, "model": fit.model, "layer": fit.layer}))
    return 0 if fit.quality == "ok" else 1


def _cmd_simulate(args: argparse.Namespace) -> int:
    from fre.engine.rate import run_rate
    from fre.io.connectome import erdos_renyi_graph, load_connectome
    from fre.offline.fit import load_fit

    if args.graph:
        graph = load_connectome(args.graph, annotations_path=args.annotations)
    else:
        graph = erdos_renyi_graph(args.n, p=args.p, n_types=2, seed=args.seed)
    fit = load_fit(args.fit)
    tables = [fit for _ in graph.type_names]
    I_ext = np.zeros(graph.n_nodes)
    I_ext[: max(1, graph.n_nodes // 10)] = args.I_ext
    state, _ = run_rate(graph, tables, n_steps=args.steps, I_ext=I_ext)
    print(json.dumps({"mean_rate": float(np.mean(state.r)), "n_nodes": graph.n_nodes}))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from fre.offline.fit import load_fit
    from fre.validation import validate_hopkins_flywire, validate_single

    if args.hopkins:
        print(json.dumps(validate_hopkins_flywire()))
        return 0
    if args.pysr_exam:
        from fre.offline.pysr_backend import admission_exam

        report = admission_exam()
        print(json.dumps(report.__dict__))
        return 0 if report.status == "passed" else 0
    fit = load_fit(args.fit)
    report = validate_single(fit)
    print(
        json.dumps(
            {
                "r2": report.r2,
                "mse": report.mse,
                "nfr4_pass": report.nfr4_pass,
                "relative_vp": report.relative_vp,
            }
        )
    )
    return 0 if report.nfr4_pass else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fre", description="FRE neuron engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="Offline f-I / LUT fit")
    fit.add_argument("--model", default="adexp")
    fit.add_argument("--type-id", default="demo")
    fit.add_argument("--I-min", dest="I_min", type=float, default=0.0)
    fit.add_argument("--I-max", dest="I_max", type=float, default=0.8)
    fit.add_argument("--n-I", dest="n_I", type=int, default=16)
    fit.add_argument("--t-total", dest="t_total", type=float, default=800.0)
    fit.add_argument("--window", type=float, default=400.0)
    fit.add_argument("--dt", type=float, default=0.05)
    fit.add_argument("--lut", action="store_true")
    fit.add_argument("--srm", action="store_true")
    fit.add_argument("--chirp", action="store_true")
    fit.add_argument("--out", default="artifacts/fit.npz")
    fit.set_defaults(func=_cmd_fit)

    sim = sub.add_parser("simulate", help="Run rate mode")
    sim.add_argument("--graph", default=None)
    sim.add_argument("--annotations", default=None)
    sim.add_argument("--fit", required=True)
    sim.add_argument("--n", type=int, default=100)
    sim.add_argument("--p", type=float, default=0.05)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--steps", type=int, default=100)
    sim.add_argument("--I-ext", dest="I_ext", type=float, default=0.4)
    sim.set_defaults(func=_cmd_simulate)

    val = sub.add_parser("validate", help="Validate a fitted F")
    val.add_argument("--fit", default=None)
    val.add_argument("--hopkins", action="store_true")
    val.add_argument("--pysr-exam", dest="pysr_exam", action="store_true")
    val.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
