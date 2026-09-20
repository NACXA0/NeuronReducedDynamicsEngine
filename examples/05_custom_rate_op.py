"""Append or replace RateOp steps. Does not write seed artifacts.

Default engine contract is type-only. Extra steps are exploration switches
(RFC-003); identity steps must match the published pipeline pointwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nrde.engine.ops import append_op, register_op, replace_op, reset_pipeline
from nrde.engine.rate import batched_apply
from nrde.fitting.fit import save_fit
from nrde.types import NPZ_SCHEMA_VERSION, FittedActivation


@dataclass
class GainOp:
    """Scale rates. g=1 is identity; an array g is a coefficient turned into a function."""

    name: str = "gain"
    g: float = 1.0

    def is_identity(self, fit: FittedActivation) -> bool:
        del fit
        return abs(float(self.g) - 1.0) <= 1e-15

    def apply(self, r, I, fit, region, xyz):
        del I, region, xyz
        if self.is_identity(fit):
            return r, 0
        return np.asarray(r, dtype=np.float64) * float(self.g), 0


@dataclass
class LinearMapOp:
    """Abstract r = clip(k I, 0, rmax). Not an ODE reduction; do not mark quality ok."""

    name: str = "linear_map"
    k: float = 20.0
    rmax: float = 100.0

    def is_identity(self, fit: FittedActivation) -> bool:
        del fit
        return False

    def apply(self, r, I, fit, region, xyz):
        del r, fit, region, xyz
        rate = np.clip(float(self.k) * np.asarray(I, dtype=np.float64), 0.0, float(self.rmax))
        return rate, 0


def _toy_fit() -> FittedActivation:
    return FittedActivation(
        schema_version=NPZ_SCHEMA_VERSION,
        type_id="pipeline_demo",
        model="lif",
        params={},
        method="linear",
        I_grid=np.array([0.0, 1.0]),
        r_grid=np.array([0.0, 10.0]),
        I_onset=0.0,
        r2=1.0,
        mse=0.0,
        quality="ok",
    )


def main() -> None:
    reset_pipeline()
    fit = _toy_fit()
    I = np.array([0.5, 0.5])
    types = np.zeros(2, dtype=np.int32)
    base, _ = batched_apply([fit], I, types)

    register_op(GainOp(g=1.0))
    append_op("gain")
    same, _ = batched_apply([fit], I, types)

    reset_pipeline()
    register_op(GainOp(g=1.1), overwrite=True)
    append_op("gain")
    scaled, _ = batched_apply([fit], I, types)

    reset_pipeline()
    register_op(LinearMapOp())
    replace_op("type", "linear_map")
    abstract, _ = batched_apply([fit], I, types)

    out = Path("/tmp/nrde_fit")
    out.mkdir(parents=True, exist_ok=True)
    save_fit(fit, out / "pipeline_demo.npz", skip_if_unchanged=False)
    print(
        {
            "base": base.tolist(),
            "identity_gain": same.tolist(),
            "scaled": scaled.tolist(),
            "abstract_linear": abstract.tolist(),
            "out": str(out / "pipeline_demo.npz"),
        }
    )
    reset_pipeline()


if __name__ == "__main__":
    main()
