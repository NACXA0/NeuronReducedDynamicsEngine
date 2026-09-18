"""TC-5.x FlyGym v2 adapter — mock Simulation, 100-step closed loop."""

from __future__ import annotations

import numpy as np
import pytest

from nrde.adapters.embodied import OpenLoopStimEnv
from nrde.adapters.flygym import MockFlyGymSim, NRDEFlyGymEnv, maybe_make_flygym_sim
from nrde.fitting.fit import fit_fi
from nrde.io.connectome import erdos_renyi_graph


@pytest.fixture(scope="module")
def small_ready():
    fit = fit_fi(
        "lif",
        type_id="type_0",
        I_min=0.0,
        I_max=0.8,
        n_I=8,
        t_total=300.0,
        window=180.0,
        dt=0.05,
        n_validate=2,
    )
    g = erdos_renyi_graph(24, p=0.08, n_types=2, seed=7, weight=0.001)
    return g, [fit, fit]


def test_tc_5_1_sense_to_current(small_ready):
    g, tables = small_ready
    sim = MockFlyGymSim(n_actuators=4)
    env = NRDEFlyGymEnv(
        sim,
        g,
        tables,
        sense_idx=np.arange(4),
        motor_idx=np.arange(4, 8),
        n_actuators=4,
    )
    obs = env.read_observation()
    current = env.sense_to_current(obs)
    assert current.shape == (g.n_nodes,)
    assert np.all(current[:4] > current[4:].mean())
    assert float(np.mean(current[:4])) > 0.0


def test_tc_5_2_motor_decode_finite(small_ready):
    g, tables = small_ready
    sim = MockFlyGymSim(n_actuators=4)
    env = NRDEFlyGymEnv(
        sim,
        g,
        tables,
        sense_idx=np.arange(3),
        motor_idx=np.arange(3, 9),
        n_actuators=4,
    )
    env.state.r[:] = 40.0  # type: ignore[attr-defined]
    action = env.spikes_to_action(env.motor_readout())
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)


@pytest.mark.flygym
def test_tc_5_3_closed_loop_100_steps(small_ready):
    g, tables = small_ready
    sim = MockFlyGymSim(n_actuators=6)
    env = NRDEFlyGymEnv(
        sim,
        g,
        tables,
        sense_idx=np.arange(5),
        motor_idx=np.arange(5, 11),
        n_actuators=6,
    )
    result = env.run(100)
    assert result["n_steps"] == 100
    assert sim.n_steps == 100
    assert result["finite"]
    assert result["nonzero"]
    assert not result["diverged"]
    assert "sense_motor_corr" in result
    assert np.isfinite(result["sense_motor_corr"])
    assert sim.last_inputs is not None
    assert sim.last_inputs.shape == (6,)


def test_open_loop_stim_env_correlation():
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.6)
    g = erdos_renyi_graph(12, p=0.1, n_types=1, seed=0, type_names=("type_0",))
    sched = np.linspace(0.1, 0.8, 40)[:, None] * np.ones((40, g.n_nodes))
    env = OpenLoopStimEnv(g, [fit], sched, n_actuators=3)
    result = env.run(40)
    assert result["finite"]
    assert not result["diverged"]


def test_open_loop_scalar_current():
    fit = fit_fi("lif", n_I=6, t_total=200.0, window=120.0, n_validate=2, I_max=0.6)
    g = erdos_renyi_graph(6, p=0.1, n_types=1, seed=0, type_names=("type_0",))
    env = OpenLoopStimEnv(g, [fit], np.array([0.4]), n_actuators=2)
    out = env.step()
    assert out["I_ext"].shape == (6,)
    assert np.allclose(out["I_ext"], 0.4)


def test_core_does_not_import_flygym():
    import nrde.engine.rate as rate

    assert "flygym" not in rate.__dict__
    assert maybe_make_flygym_sim() is MockFlyGymSim or callable(maybe_make_flygym_sim)
