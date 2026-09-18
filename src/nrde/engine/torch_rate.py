"""PyTorch CSR rate engine (CUDA via cuSPARSE; CPU for parity)."""

from __future__ import annotations

from typing import Any

import numpy as np

from nrde.engine.sparse import get_csr, get_type_indices, prepare_graph
from nrde.types import FittedActivation, GraphData, RateState


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "backend='torch' requires pip install 'neuron-reduced-dynamics-engine[torch]'"
        ) from exc
    return torch


def _device(torch: Any, device: str | None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _interp1d(x, xp, fp):
    """Linear interp matching ``np.interp`` for increasing ``xp`` (clamped)."""
    torch = _require_torch()
    x = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    denom = (x1 - x0).clamp_min(1e-12)
    return y0 + (x - x0) / denom * (y1 - y0)


def _eval_rate_torch(I, I_grid, r_grid, I_onset):
    """Match ``FittedActivation.eval_rate`` (float64 onset / interp semantics)."""
    torch = _require_torch()
    oor = ((I < I_grid[0]) | (I > I_grid[-1])).sum()
    I_clip = I.clamp(I_grid[0], I_grid[-1])
    r = _interp1d(I_clip, I_grid, r_grid)
    r = torch.where(I_clip < I_onset, torch.zeros_like(r), r)
    r = r.clamp(min=0.0)
    if r_grid.numel():
        r = torch.minimum(r, r_grid.max())
    return r, oor


def _csr_to_torch(graph: GraphData, device):
    torch = _require_torch()
    csr = get_csr(graph)
    n = int(graph.n_nodes)
    crow = torch.as_tensor(np.ascontiguousarray(csr.indptr, dtype=np.int64), device=device)
    col = torch.as_tensor(np.ascontiguousarray(csr.indices, dtype=np.int64), device=device)
    # SpMV stays float32 (cuSPARSE); activation tables use float64 below for numpy parity.
    val = torch.as_tensor(np.ascontiguousarray(csr.data, dtype=np.float32), device=device)
    return torch.sparse_csr_tensor(crow, col, val, size=(n, n), device=device, dtype=torch.float32)


def prepare_torch_graph(
    graph: GraphData,
    tables: list[FittedActivation],
    device: str | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    dev = _device(torch, device)
    token = (id(graph.edge_index), id(graph.edge_weight), graph.n_nodes, str(dev), "f64-act")
    cached = graph._torch
    if isinstance(cached, dict) and cached.get("token") == token:
        return cached
    prepare_graph(graph, n_types=len(tables))
    idx_np = get_type_indices(graph, len(tables))
    bundle: dict[str, Any] = {
        "token": token,
        "device": dev,
        "csr": _csr_to_torch(graph, dev),
        "idx": [torch.as_tensor(i, dtype=torch.int64, device=dev) for i in idx_np],
        "I_grid": [],
        "r_grid": [],
        "I_onset": [],
    }
    for table in tables:
        # float64: float32 I_onset rounding breaks ``I < I_onset`` vs numpy at grid knots.
        bundle["I_grid"].append(
            torch.as_tensor(np.ascontiguousarray(table.I_grid, dtype=np.float64), device=dev)
        )
        bundle["r_grid"].append(
            torch.as_tensor(np.ascontiguousarray(table.r_grid, dtype=np.float64), device=dev)
        )
        bundle["I_onset"].append(torch.tensor(float(table.I_onset), dtype=torch.float64, device=dev))
    graph._torch = bundle
    return bundle


def step_rate_torch(
    r: Any,
    I_ext: Any,
    bundle: dict[str, Any],
    oor: Any,
) -> tuple[Any, Any]:
    torch = _require_torch()
    csr = bundle["csr"]
    # SpMV in float32; promote before F so onset/interp match numpy float64.
    I_syn = torch.sparse.mm(csr, r.to(dtype=torch.float32).unsqueeze(1)).squeeze(1).to(dtype=torch.float64)
    I_total = I_syn + I_ext.to(dtype=torch.float64)
    r_new = torch.zeros(r.shape, dtype=torch.float64, device=r.device)
    for t, idx in enumerate(bundle["idx"]):
        if idx.numel() == 0:
            continue
        rt, n_oor = _eval_rate_torch(
            I_total[idx],
            bundle["I_grid"][t],
            bundle["r_grid"][t],
            bundle["I_onset"][t],
        )
        r_new[idx] = rt
        oor = oor + n_oor
    return r_new, oor


def run_rate_torch(
    graph: GraphData,
    tables: list[FittedActivation],
    n_steps: int,
    I_ext: np.ndarray | None = None,
    dt: float = 1.0,
    r0: float = 0.0,
    record_trace: bool = True,
    device: str | None = None,
) -> tuple[RateState, np.ndarray]:
    del dt
    torch = _require_torch()
    bundle = prepare_torch_graph(graph, tables, device=device)
    dev = bundle["device"]
    I_np = np.zeros(graph.n_nodes, dtype=np.float64) if I_ext is None else np.asarray(I_ext, dtype=np.float64)
    I_t = torch.as_tensor(np.ascontiguousarray(I_np), device=dev)
    r = torch.full((graph.n_nodes,), float(r0), dtype=torch.float64, device=dev)
    oor = torch.zeros((), dtype=torch.int64, device=dev)
    trace_np = np.zeros((n_steps if record_trace else 0, graph.n_nodes), dtype=np.float64)
    for t in range(n_steps):
        r, oor = step_rate_torch(r, I_t, bundle, oor)
        if record_trace:
            trace_np[t] = r.detach().cpu().numpy()
    r_host = r.detach().cpu().numpy().astype(np.float64, copy=False)
    return RateState(r=r_host, out_of_range=int(oor.detach().cpu().item())), trace_np
