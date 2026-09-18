#!/usr/bin/env python3
"""A1: render demo frames / placeholder GIF (D21). Headless by default (Q6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import nrde


def _write_placeholder_gif(path: Path, n_frames: int, seed: int) -> None:
    """Minimal GIF without MuJoCo: pulsing grayscale frames from demo actions."""
    try:
        import imageio.v2 as imageio  # type: ignore
    except ImportError:
        # Fallback: write NPZ of frames + note (CI without imageio).
        rng = np.random.default_rng(seed)
        frames = rng.random((min(n_frames, 24), 32, 32))
        np.savez_compressed(path.with_suffix(".npz"), frames=frames)
        path.with_suffix(".txt").write_text(
            "imageio not installed; wrote frames npz instead of GIF\n",
            encoding="utf-8",
        )
        return

    rng = np.random.default_rng(seed)
    frames = []
    for i in range(min(n_frames, 48)):
        level = 0.3 + 0.5 * ((i + seed) % 20) / 20.0
        img = (np.clip(level + 0.05 * rng.standard_normal((64, 64)), 0, 1) * 255).astype(np.uint8)
        frames.append(np.stack([img, img, img], axis=-1))
    imageio.mimsave(path, frames, duration=0.05)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="flygym-demo-v01")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("assets/demo"))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    result = nrde.demo("flygym", steps=args.steps, headless=args.headless)
    stamp = {
        "nrde_version": nrde.__version__,
        "preset": args.preset,
        "steps": args.steps,
        "seed": args.seed,
        "n_steps": result.get("n_steps"),
        "finite": result.get("finite"),
        "sense_motor_corr": result.get("sense_motor_corr"),
    }
    (args.out / "last_run.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    (args.out / "WATERMARK.txt").write_text(
        f"NRDE v{nrde.__version__}  preset={args.preset}  seed={args.seed}  steps={args.steps}\n",
        encoding="utf-8",
    )
    gif = args.out / f"demo_seed{args.seed}.gif"
    _write_placeholder_gif(gif, n_frames=args.steps, seed=args.seed)
    print(json.dumps({"out": str(args.out), "gif": str(gif), **stamp}))
    return 0 if result.get("finite") else 1


if __name__ == "__main__":
    raise SystemExit(main())
