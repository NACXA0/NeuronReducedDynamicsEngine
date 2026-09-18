"""EmbodiedEnv ABC (DD-5 / RFC-001 D4). FlyGym is one backend; no custom body in v0.1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from nrde.engine.rate import step_rate
from nrde.types import FittedActivation, GraphData, RateState


class EmbodiedEnv(ABC):
    """Hooks only. v0.1 claims interface correctness, not meaningful behavior."""

    @abstractmethod
    def sense_to_current(self, obs: dict[str, Any]) -> np.ndarray: ...

    @abstractmethod
    def spikes_to_action(self, motor_out: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def read_observation(self) -> dict[str, Any]: ...

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None: ...

    def motor_readout(self) -> np.ndarray:
        raise NotImplementedError

    def neural_step(self, I_ext: np.ndarray) -> None:
        raise NotImplementedError

    def step(self) -> dict[str, Any]:
        obs = self.read_observation()
        I_ext = self.sense_to_current(obs)
        self.neural_step(I_ext)
        action = self.spikes_to_action(self.motor_readout())
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Non-finite actuator command")
        self.apply_action(action)
        return {"obs": obs, "action": action, "I_ext": I_ext}

    def run(self, n_steps: int) -> dict[str, Any]:
        actions = []
        currents = []
        for _ in range(n_steps):
            result = self.step()
            actions.append(result["action"])
            currents.append(result["I_ext"])
        stacked = np.stack(actions, axis=0)
        I = np.stack(currents, axis=0)
        sense = I.mean(axis=1)
        motor = np.sqrt((stacked**2).mean(axis=1))
        corr = 0.0
        if sense.std() > 1e-12 and motor.std() > 1e-12:
            corr = float(np.corrcoef(sense, motor)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        return {
            "n_steps": n_steps,
            "actions": stacked,
            "finite": bool(np.all(np.isfinite(stacked))),
            "nonzero": bool(np.any(np.abs(stacked) > 1e-8)),
            "sense_motor_corr": corr,
            "diverged": bool(not np.all(np.isfinite(stacked))),
        }


class OpenLoopStimEnv(EmbodiedEnv):
    """No body: inject a prescribed current sequence (validation)."""

    def __init__(
        self,
        graph: GraphData,
        tables: list[FittedActivation],
        I_schedule: np.ndarray,
        n_actuators: int = 4,
        motor_idx: np.ndarray | None = None,
    ):
        self.graph = graph
        self.tables = tables
        self.I_schedule = np.asarray(I_schedule, dtype=np.float64)
        self.t = 0
        self.n_actuators = n_actuators
        self.motor_idx = (
            np.arange(n_actuators)
            if motor_idx is None
            else np.asarray(motor_idx, dtype=int)
        )
        self.state: RateState = RateState(r=np.zeros(graph.n_nodes, dtype=np.float64))
        self.last_obs: dict[str, Any] = {"t": 0}

    def read_observation(self) -> dict[str, Any]:
        k = min(self.t, self.I_schedule.shape[0] - 1)
        return {"t": self.t, "I": self.I_schedule[k]}

    def sense_to_current(self, obs: dict[str, Any]) -> np.ndarray:
        I = np.asarray(obs["I"], dtype=np.float64)
        if I.ndim == 0:
            out = np.zeros(self.graph.n_nodes)
            out[:] = float(I)
            return out
        return I

    def neural_step(self, I_ext: np.ndarray) -> None:
        self.state = step_rate(self.state, self.graph, self.tables, I_ext)

    def motor_readout(self) -> np.ndarray:
        return self.state.r

    def spikes_to_action(self, motor_out: np.ndarray) -> np.ndarray:
        sl = motor_out[self.motor_idx]
        if sl.size < self.n_actuators:
            sl = np.resize(sl, self.n_actuators)
        return np.tanh(sl[: self.n_actuators] * 0.05)

    def apply_action(self, action: np.ndarray) -> None:
        del action
        self.t += 1
