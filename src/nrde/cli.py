"""CLI: demo / run / offline (+ legacy fit|simulate|validate aliases). RFC-002 §3.5."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _cmd_fit(args: argparse.Namespace) -> int:
    from nrde.fitting.artifacts import default_artifact_path
    from nrde.fitting.fit import attach_lut, fit_fi, fit_spike_lut, save_fit

    rounds = None
    if getattr(args, "layers", None):
        from nrde.fitting.iterate import fit_layers

        fit, rounds = fit_layers(
            args.layers,
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
    elif getattr(args, "auto", False):
        from nrde.fitting.iterate import fit_iterative

        fit, rounds = fit_iterative(
            args.model,
            type_id=args.type_id,
            I_min=args.I_min,
            I_max=args.I_max,
            n_I=args.n_I,
            t_total=args.t_total,
            window=args.window,
            dt=args.dt,
            chirp=args.chirp,
            max_layer=args.max_layer,
            prefer=args.prefer,
        )
    else:
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
            from nrde.fitting.srm import attach_srm, fit_srm_kernels

            fit = attach_srm(fit, fit_srm_kernels(args.model, t_kernel=40.0, dt=0.1))
    out = args.out or str(default_artifact_path(fit))
    path = save_fit(fit, out, skip_if_unchanged=not getattr(args, "force", False))
    payload = {
        "out": str(path),
        "r2": fit.r2,
        "quality": fit.quality,
        "model": fit.model,
        "layer": fit.layer,
        "type_id": fit.type_id,
        "config_hash": fit.config_hash,
    }
    if rounds is not None:
        payload["rounds"] = rounds
        payload["fallback_ode"] = bool(fit.fallback_ode)
        payload["prefer"] = getattr(args, "prefer", "efficiency")
    print(json.dumps(payload))
    return 0 if fit.quality == "ok" else 1


def _cmd_simulate(args: argparse.Namespace) -> int:
    from nrde.binding import bind_tables, bound_type_ids, parse_fit_map, parse_type_names
    from nrde.engine.rate import run_rate
    from nrde.fitting.fit import load_fit
    from nrde.io.connectome import erdos_renyi_graph, load_connectome
    from nrde.progress import progress_session

    fit_specs = getattr(args, "fit_map", None)
    if not args.fit and not fit_specs:
        print("simulate needs --fit and/or --fit-map TYPE=PATH", file=sys.stderr)
        return 2
    try:
        type_names = parse_type_names(getattr(args, "type_names", None))
        path_by_type = parse_fit_map(fit_specs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.graph and type_names is not None:
        print("--type-names only applies to the random graph", file=sys.stderr)
        return 2

    with progress_session():
        if args.graph:
            graph = load_connectome(args.graph, annotations_path=args.annotations)
        else:
            names = type_names or ("type_0", "type_1")
            graph = erdos_renyi_graph(
                args.n,
                p=args.p,
                n_types=len(names),
                seed=args.seed,
                type_names=names,
            )
        by_name = {name: load_fit(path) for name, path in path_by_type.items()}
        fallback = load_fit(args.fit) if args.fit else None
        try:
            tables = bind_tables(graph, by_name, fallback=fallback)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        I_ext = np.zeros(graph.n_nodes)
        I_ext[: max(1, graph.n_nodes // 10)] = args.I_ext
        state, _ = run_rate(graph, tables, n_steps=args.steps, I_ext=I_ext, record_trace=False)
    print(
        json.dumps(
            {
                "mean_rate": float(np.mean(state.r)),
                "n_nodes": graph.n_nodes,
                "bound": bound_type_ids(graph, tables),
            }
        )
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from nrde.fitting.fit import load_fit
    from nrde.validation import validate_hopkins_flywire, validate_single

    if args.hopkins:
        print(json.dumps(validate_hopkins_flywire()))
        return 0
    if args.pysr_exam:
        from nrde.fitting.pysr_backend import admission_exam

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
                "nfr5_pass": report.nfr5_pass,
            }
        )
    )
    return 0 if report.nfr4_pass else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    from nrde.presets import demo

    result = demo(
        args.name,
        steps=args.steps,
        headless=args.headless,
        visual=args.visual,
        config=args.config,
    )
    print(
        json.dumps(
            {
                "steps": result.get("n_steps"),
                "finite": result.get("finite"),
                "nonzero": result.get("nonzero"),
                "sense_motor_corr": result.get("sense_motor_corr"),
            }
        )
    )
    ok = bool(result.get("finite")) and int(result.get("n_steps", 0)) == int(args.steps)
    return 0 if ok else 1


def _fetch_notice(report: dict) -> str:
    summary = report.get("summary") or {}
    verified = int(summary.get("verified") or 0)
    downloaded = int(summary.get("downloaded") or 0)
    tier = report.get("tier")
    if not report.get("ok"):
        return f"{tier}: 校验失败"
    if downloaded:
        return f"{tier}: {verified} 已校验，{downloaded} 已下载"
    return f"{tier}: {verified} 已校验，未下载"


def _cmd_fetch(args: argparse.Namespace) -> int:
    from nrde.fetch import fetch_preset

    report = fetch_preset(
        args.preset,
        tier=args.tier,
        refresh=args.refresh,
        manifest_path=args.manifest,
        skip_weights=args.skip_weights,
    )
    print(_fetch_notice(report), file=sys.stderr)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


def _add_fit_parser(sub: argparse._SubParsersAction, name: str = "fit") -> None:
    fit = sub.add_parser(name, help="Offline f-I / LUT fit")
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
    fit.add_argument("--auto", action="store_true", help="Climb layers until the gates pass")
    fit.add_argument(
        "--prefer",
        choices=("efficiency", "neutral", "fidelity"),
        default="efficiency",
        help="With --auto: stop early (efficiency, default), upgrade only if VP drops (neutral), or climb to --max-layer (fidelity)",
    )
    fit.add_argument(
        "--max-layer",
        dest="max_layer",
        choices=("L0", "L1", "L2", "L3"),
        default="L2",
        help="Ceiling for --auto (L3 records deferred_v0.2, it does not train FNO)",
    )
    fit.add_argument(
        "--layers",
        choices=("L0", "L1", "L2", "L1+L2"),
        default=None,
        help="One-shot mix; skips --auto densify/climb",
    )
    fit.add_argument("--force", action="store_true", help="Rewrite even if config_hash matches")
    fit.add_argument(
        "--out",
        default=None,
        help="Output npz (default: artifacts/{type_id}.npz — type-keyed)",
    )
    fit.set_defaults(func=_cmd_fit)


def _add_simulate_parser(sub: argparse._SubParsersAction, name: str = "simulate") -> None:
    sim = sub.add_parser(name, help="Run rate-mode circuit")
    sim.add_argument("--graph", default=None)
    sim.add_argument("--annotations", default=None)
    sim.add_argument(
        "--fit",
        default=None,
        help="Fallback F table for types without --fit-map (README smoke path)",
    )
    sim.add_argument(
        "--fit-map",
        action="append",
        default=None,
        metavar="TYPE=PATH",
        help="Bind a type name to an npz; repeatable. Unmapped types use --fit",
    )
    sim.add_argument(
        "--type-names",
        dest="type_names",
        default=None,
        help="Comma-separated type names for the random graph (default: type_0,type_1)",
    )
    sim.add_argument("--n", type=int, default=100)
    sim.add_argument("--p", type=float, default=0.05)
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--steps", type=int, default=100)
    sim.add_argument("--I-ext", dest="I_ext", type=float, default=0.4)
    sim.set_defaults(func=_cmd_simulate)


def _add_validate_parser(sub: argparse._SubParsersAction, name: str = "validate") -> None:
    val = sub.add_parser(name, help="Validate a fitted F")
    val.add_argument("--fit", default=None)
    val.add_argument("--hopkins", action="store_true")
    val.add_argument("--pysr-exam", dest="pysr_exam", action="store_true")
    val.set_defaults(func=_cmd_validate)


def _fetch_tier_choices() -> tuple[str, ...]:
    fallback = ("smoke", "seed", "pretrained", "full")
    try:
        from nrde.fetch import load_manifest

        names = tuple((load_manifest().get("tiers") or {}).keys())
    except Exception:  # noqa: BLE001 — parser must still build without a manifest
        return fallback
    return names or fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nrde",
        description="NRDE — product surfaces A1/A2/B/C (RFC-002 v1.1)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # A2 demo
    demo_p = sub.add_parser("demo", help="A2: run a preset embodied demo")
    demo_p.add_argument("name", nargs="?", default="flygym", help="Preset name (default: flygym)")
    demo_p.add_argument("--steps", type=int, default=100)
    demo_p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    demo_p.add_argument("--visual", default=None, help='e.g. "left_light_on"')
    demo_p.add_argument("--config", default=None, help="Path to flygym_demo.yaml")
    demo_p.set_defaults(func=_cmd_demo)

    # A2 data
    fetch_p = sub.add_parser("fetch", help="A2: fetch/verify a data preset via manifest")
    fetch_p.add_argument("preset", nargs="?", default="fly", help="Preset name (default: fly)")
    fetch_p.add_argument(
        "--tier",
        choices=_fetch_tier_choices(),
        default="seed",
        help="smoke=seed hashes; seed=artifacts; pretrained=datasets/fly; full=+connectome download",
    )
    fetch_p.add_argument("--refresh", action="store_true")
    fetch_p.add_argument("--manifest", default=None, help="Override manifest YAML path")
    fetch_p.add_argument(
        "--skip-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For tier=full, skip ~1.1GB weights Feather (default: true)",
    )
    fetch_p.set_defaults(func=_cmd_fetch)

    # B-layer
    run_p = sub.add_parser("run", help="B-layer: engine entry")
    run_sub = run_p.add_subparsers(dest="run_cmd", required=True)
    _add_simulate_parser(run_sub, "simulate")

    # C-layer
    off = sub.add_parser("offline", help="C-layer: fitting pipeline (impl: nrde.fitting)")
    off_sub = off.add_subparsers(dest="offline_cmd", required=True)
    _add_fit_parser(off_sub, "fit")
    _add_validate_parser(off_sub, "validate")

    # Legacy aliases (pre-RFC-002)
    _add_fit_parser(sub, "fit")
    _add_simulate_parser(sub, "simulate")
    _add_validate_parser(sub, "validate")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Programmatic B-layer entry: ``nrde.run(["simulate", ...])``."""
    return main(["run", *(argv or [])])


def offline(argv: list[str] | None = None) -> int:
    """Programmatic C-layer entry: ``nrde.offline(["fit", ...])``."""
    return main(["offline", *(argv or [])])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
