"""Same kernel as the fly smoke path; only the type names change.

Not a population model. After ``nrde run simulate --fit aCC_l0.npz``, this
shows that graph type labels and F tables are arguments, not the engine.
"""

from __future__ import annotations

import json

import numpy as np

from nrde.binding import bind_tables, bound_type_ids
from nrde.engine.rate import run_rate
from nrde.fitting.fit import fit_fi
from nrde.io.connectome import erdos_renyi_graph


def main() -> dict:
    prey = fit_fi(
        "explif",
        type_id="prey",
        n_I=8,
        t_total=250.0,
        window=150.0,
        I_max=0.8,
        n_validate=2,
    )
    predator = fit_fi(
        "lif",
        type_id="predator",
        n_I=8,
        t_total=250.0,
        window=150.0,
        I_max=0.8,
        n_validate=2,
    )
    graph = erdos_renyi_graph(
        40,
        p=0.08,
        n_types=2,
        seed=0,
        weight=0.0008,
        type_names=("prey", "predator"),
    )
    tables = bind_tables(graph, {"prey": prey, "predator": predator})
    I_ext = np.full(graph.n_nodes, 0.35)
    state, trace = run_rate(graph, tables, n_steps=30, I_ext=I_ext)
    prey_mask = graph.node_type == 0
    predator_mask = graph.node_type == 1
    payload = {
        "mean_rate": float(np.mean(state.r)),
        "n_nodes": graph.n_nodes,
        "bound": bound_type_ids(graph, tables),
        "prey_mean": float(np.mean(state.r[prey_mask])) if np.any(prey_mask) else 0.0,
        "predator_mean": float(np.mean(state.r[predator_mask])) if np.any(predator_mask) else 0.0,
    }
    print(json.dumps(payload))
    if not np.all(np.isfinite(trace)):
        raise RuntimeError("non-finite rate trace")
    return payload


if __name__ == "__main__":
    main()
