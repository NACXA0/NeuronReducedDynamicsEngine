from __future__ import annotations

import numpy as np

from fre.adapters.flygym import FREFlyGymEnv, MockFlyGymSim, maybe_make_flygym_sim
from fre.io.connectome import erdos_renyi_graph
from fre.offline.fit import fit_fi


def main() -> None:
    fit = fit_fi("explif", n_I=10, t_total=400.0, window=250.0, I_max=0.7)
    g = erdos_renyi_graph(40, p=0.05, n_types=2, seed=0, weight=0.001)
    n_act = 6
    Factory = maybe_make_flygym_sim()
    sim = Factory(n_actuators=n_act) if Factory is MockFlyGymSim else MockFlyGymSim(n_actuators=n_act)
    env = FREFlyGymEnv(
        sim,
        g,
        [fit, fit],
        sense_idx=np.arange(8),
        motor_idx=np.arange(8, 8 + n_act),
        n_actuators=n_act,
    )
    result = env.run(100)
    print(
        {
            "steps": result["n_steps"],
            "finite": result["finite"],
            "nonzero": result["nonzero"],
            "sense_motor_corr": result["sense_motor_corr"],
            "action_rms": float(np.sqrt(np.mean(result["actions"] ** 2))),
            "backend": type(sim).__name__,
        }
    )
    assert result["finite"] and result["n_steps"] == 100


if __name__ == "__main__":
    main()
