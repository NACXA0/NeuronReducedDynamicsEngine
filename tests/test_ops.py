"""Pluggable RateOp pipeline. Default order is type, place, modulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from nrde.engine.ops import (
    append_op,
    get_pipeline,
    list_ops,
    pipeline_allows_torch,
    register_op,
    replace_op,
    reset_pipeline,
)
from nrde.engine.rate import batched_apply
from nrde.types import NPZ_SCHEMA_VERSION, FittedActivation, PlaceMod


@pytest.fixture(autouse=True)
def _reset_ops():
    reset_pipeline()
    yield
    reset_pipeline()


def _fit(**kwargs) -> FittedActivation:
    fit = FittedActivation(
        schema_version=NPZ_SCHEMA_VERSION,
        type_id="t",
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
    return fit if not kwargs else FittedActivation(**{**fit.__dict__, **kwargs})


@dataclass
class GainOp:
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
    name: str = "linear_map"
    k: float = 8.0

    def is_identity(self, fit: FittedActivation) -> bool:
        del fit
        return False

    def apply(self, r, I, fit, region, xyz):
        del r, fit, region, xyz
        return float(self.k) * np.asarray(I, dtype=np.float64), 0


def test_default_pipeline_matches_eval_rate():
    fit = _fit()
    I = np.array([0.25, 0.75])
    got, _ = batched_apply([fit], I, np.zeros(2, dtype=np.int32))
    want, _ = fit.eval_rate(I)
    np.testing.assert_allclose(got, want)
    assert get_pipeline() == ("type", "place", "modulator")
    assert pipeline_allows_torch([fit])


def test_append_identity_does_not_change_rates():
    fit = _fit()
    I = np.array([0.4, 0.6])
    before, _ = batched_apply([fit], I, np.zeros(2, dtype=np.int32))
    register_op(GainOp(g=1.0))
    append_op("gain")
    after, _ = batched_apply([fit], I, np.zeros(2, dtype=np.int32))
    np.testing.assert_allclose(after, before)
    assert not pipeline_allows_torch([fit])


def test_gain_op_scales_only_rate():
    fit = _fit()
    I = np.array([0.5, 0.5])
    base, _ = batched_apply([fit], I, np.zeros(2, dtype=np.int32))
    register_op(GainOp(g=1.5))
    append_op("gain")
    scaled, _ = batched_apply([fit], I, np.zeros(2, dtype=np.int32))
    np.testing.assert_allclose(scaled, base * 1.5)


def test_replace_type_uses_custom_map():
    fit = _fit()
    I = np.array([0.2, 0.8])
    register_op(LinearMapOp(k=8.0))
    replace_op("type", "linear_map")
    got, _ = batched_apply([fit], I, np.zeros(2, dtype=np.int32))
    np.testing.assert_allclose(got, 8.0 * I)


def test_unknown_op_lists_known_names():
    with pytest.raises(ValueError, match="place"):
        append_op("not_a_step")
    assert "type" in list_ops()


def test_reset_drops_user_ops():
    register_op(GainOp())
    append_op("gain")
    reset_pipeline()
    assert get_pipeline() == ("type", "place", "modulator")
    assert "gain" not in list_ops()
    fit = _fit(place=PlaceMod(region_gain={"MB": 0.0}))
    assert pipeline_allows_torch([fit])
