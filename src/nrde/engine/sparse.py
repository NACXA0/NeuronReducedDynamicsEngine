"""Sparse gather-scatter current aggregation."""

from __future__ import annotations

import numpy as np


def sparse_matvec(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    x: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """I_syn[post] += w * x[pre]."""
    pre = edge_index[0]
    post = edge_index[1]
    out = np.zeros(n_nodes, dtype=np.float64)
    if pre.size == 0:
        return out
    np.add.at(out, post, edge_weight.astype(np.float64) * x.astype(np.float64)[pre])
    return out
