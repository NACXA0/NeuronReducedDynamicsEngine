"""FlyGym v2 adapter (FR-5, DD-5). Core engine never imports flygym."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from fre.adapters.embodied import EmbodiedEnv
from fre.engine.rate import step_rate
from fre.engine.spike import init_spike_state, step_spike_lut
from fre.types import FittedActivation, GraphData, RateState, SpikeState


class FlyGymSimulation(Protocol):
    def step(self) -> None: ...

    def get_raw_vision(self, fly_name: str = "fly") -> np.ndarray: ...

    def set_actuator_inputs(self, fly_name: str, actuator_type: str, inputs: np.ndarray) -> None: ...


@dataclass
class LinearSenseEncoder:
    sense_idx: np.ndarray
    gain: float = 1.0
    bias: float = 0.35

    def __call__(self, n_nodes: int, vision: np.ndarray) -> np.ndarray:
        current = np.full(n_nodes, self.bias, dtype=np.float64)
        level = float(np.mean(vision)) * self.gain
        current[self.sense_idx] += level
        return current


@dataclass
class LowpassMotorDecoder:
    motor_idx: np.ndarray
    n_actuators: int
    window: int = 20
    gain: float = 0.05
    history: list[np.ndarray] = field(default_factory=list)

    def __call__(self, readout: np.ndarray) -> np.ndarray:
        self.history.append(np.asarray(readout, dtype=np.float64))
        self.history = self.history[-self.window :]
        stacked = np.mean(np.stack(self.history, axis=0), axis=0)
        motor = stacked[self.motor_idx]
        if motor.size == 0:
            return np.zeros(self.n_actuators, dtype=np.float64)
        if motor.size < self.n_actuators:
            motor = np.resize(motor, self.n_actuators)
        action = np.tanh(motor * self.gain)
        return action[: self.n_actuators]


class MockFlyGymSim:
    """Stand-in for FlyGym v2 Simulation when the extra is not installed."""

    def __init__(self, n_actuators: int = 6, vision_shape: tuple[int, ...] = (2, 8, 8)):
        self.n_actuators = n_actuators
        self._vision = np.ones(vision_shape, dtype=np.float64) * 0.6
        self.last_inputs: np.ndarray | None = None
        self.n_steps = 0

    def get_raw_vision(self, fly_name: str = "fly") -> np.ndarray:
        del fly_name
        return self._vision

    def set_actuator_inputs(self, fly_name: str, actuator_type: str, inputs: np.ndarray) -> None:
        del fly_name, actuator_type
        self.last_inputs = np.asarray(inputs, dtype=np.float64)

    def step(self) -> None:
        self.n_steps += 1
        level = 0.3 + 0.5 * ((self.n_steps % 20) / 20.0)
        self._vision = np.ones(self._vision.shape, dtype=np.float64) * level


class FREFlyGymEnv(EmbodiedEnv):
    """FlyGym v2 backend of EmbodiedEnv. Interface demo only (RFC-001 D5)."""

    def __init__(
        self,
        sim: Any,
        graph: GraphData,
        tables: list[FittedActivation],
        sense_idx: np.ndarray,
        motor_idx: np.ndarray,
        n_actuators: int,
        mode: str = "rate",
        fly_name: str = "fly",
        actuator_type: str = "joints",
        sense_encoder: Callable[..., np.ndarray] | None = None,
        motor_decoder: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.sim = sim
        self.graph = graph
        self.tables = tables
        self.mode = mode
        self.fly_name = fly_name
        self.actuator_type = actuator_type
        self.sense_encoder = sense_encoder or LinearSenseEncoder(np.asarray(sense_idx, dtype=int))
        self.motor_decoder = motor_decoder or LowpassMotorDecoder(
            np.asarray(motor_idx, dtype=int),
            n_actuators=n_actuators,
        )
        self.motor_idx = np.asarray(motor_idx, dtype=int)
        if mode == "rate":
            self.state: RateState | SpikeState = RateState(
                r=np.zeros(graph.n_nodes, dtype=np.float64)
            )
        else:
            self.state = init_spike_state(graph, tables)

    def read_observation(self) -> dict[str, Any]:
        vision = self.sim.get_raw_vision(self.fly_name)
        return {"vision": np.asarray(vision)}

    def sense_to_current(self, obs: dict[str, Any]) -> np.ndarray:
        return self.sense_encoder(self.graph.n_nodes, obs["vision"])

    def motor_readout(self) -> np.ndarray:
        if isinstance(self.state, RateState):
            return self.state.r
        return self.state.spikes.astype(np.float64)

    def spikes_to_action(self, motor_out: np.ndarray) -> np.ndarray:
        return np.asarray(self.motor_decoder(motor_out), dtype=np.float64)

    def neural_step(self, I_ext: np.ndarray) -> None:
        if self.mode == "rate":
            self.state = step_rate(self.state, self.graph, self.tables, I_ext)  # type: ignore[arg-type]
        else:
            self.state = step_spike_lut(self.state, self.graph, self.tables, I_ext)  # type: ignore[arg-type]

    def apply_action(self, action: np.ndarray) -> None:
        self.sim.set_actuator_inputs(self.fly_name, self.actuator_type, action)
        self.sim.step()


def maybe_make_flygym_sim() -> Any:
    try:
        from flygym import Simulation  # type: ignore

        return Simulation
    except Exception:
        return MockFlyGymSim
