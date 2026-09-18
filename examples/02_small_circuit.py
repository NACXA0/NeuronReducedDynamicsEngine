from __future__ import annotations

import numpy as np

from fre.binding import tables_for_graph
from fre.calibration import apply_alpha_scale, apply_shiu_weights, calibrate_alpha
from fre.engine.rate import run_rate
from fre.io.connectome import erdos_renyi_graph
from fre.offline.fit import fit_fi
from fre.validation import uniform_lif_ablation


def main() -> None:
    fit_exc = fit_fi("explif", type_id="type_0", n_I=12, t_total=500.0, window=300.0, I_max=0.8)
    fit_inh = fit_fi("lif", type_id="type_1", n_I=12, t_total=500.0, window=300.0, I_max=0.8)
    g = erdos_renyi_graph(
        200,
        p=0.03,
        n_types=2,
        seed=0,
        weight=0.0008,
        type_names=("type_0", "type_1"),
    )
    apply_shiu_weights(g)
    tables = tables_for_graph(g, {"type_0": fit_exc, "type_1": fit_inh})
    I_ext = np.full(g.n_nodes, 0.32)
    I_ext[g.node_type == 0] = 0.38
    scale, mse = calibrate_alpha(g, tables, target_rates=np.full(g.n_nodes, 25.0), I_ext=I_ext, n_steps=25)
    apply_alpha_scale(g, scale)
    state, trace = run_rate(g, tables, n_steps=200, I_ext=I_ext)
    abl = uniform_lif_ablation(g, tables, I_ext, n_steps=80)
    print(
        {
            "alpha_scale": scale,
            "cal_mse": mse,
            "mean_rate": float(np.mean(state.r)),
            "type0_mean": float(np.mean(state.r[g.node_type == 0])),
            "type1_mean": float(np.mean(state.r[g.node_type == 1])),
            "ablation": abl,
            "out_of_range": state.out_of_range,
        }
    )
    assert np.all(np.isfinite(trace))


if __name__ == "__main__":
    main()
