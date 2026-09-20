"""PyTorch CSR rate engine (CUDA via cuSPARSE; CPU for parity)."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from nrde.engine.sparse import get_csr, get_type_indices, prepare_graph
from nrde.progress import get_progress, iter_progress
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


# Scratch left free for one activation chunk (index + a handful of float64 gathers).
_ACTIVATION_SCRATCH = 512 << 20
# Default chunk: about 1e6 nodes, ~80 MiB of temporaries instead of one buffer per node.
_ACTIVATION_CHUNK = 1 << 20


def gpu_resident_bytes(n_rows: int, nnz: int, n_cols: int) -> int:
    """Bytes to keep ``n_rows`` of a CSR on device, plus one activation chunk.

    The interpolation ``fp[idx]`` materialises a float64 vector the size of the
    index. That temporary is chunked, so it is a fixed scratch, not ``n_rows``.
    """
    if n_rows <= 0:
        return 0
    return (
        (n_rows + 1) * 8
        + int(nnz) * 12
        + int(n_cols) * 12
        + n_rows * 40
        + _ACTIVATION_SCRATCH
    )


def plan_gpu_rows(indptr: np.ndarray, n_nodes: int, budget: int) -> int:
    """How many leading CSR rows fit in ``budget`` bytes. The rest stay in RAM.

    Prefer VRAM. A full graph that fits does not get padded out to fill the card.
    """
    n_nodes = int(n_nodes)
    if n_nodes <= 0 or int(budget) <= 0:
        return 0
    ptr = np.asarray(indptr, dtype=np.int64)
    if gpu_resident_bytes(n_nodes, int(ptr[n_nodes]), n_nodes) <= int(budget):
        return n_nodes
    lo = 0
    hi = n_nodes
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if gpu_resident_bytes(mid, int(ptr[mid]), n_nodes) <= int(budget):
            lo = mid
        else:
            hi = mid - 1
    return lo


def cuda_vram_budget(torch: Any) -> int:
    """Free VRAM minus a reserve, so a fit does not pin the card at 100%."""
    free, _total = torch.cuda.mem_get_info()
    reserve = max(256 << 20, int(free) // 10)
    return max(int(free) - reserve, 0)


def _report_residency(n_fast: int, n_nodes: int) -> None:
    progress = get_progress()
    if not progress.enabled:
        return
    n_fast = int(n_fast)
    n_nodes = int(n_nodes)
    if n_fast >= n_nodes:
        line = f"residency  {n_nodes:,}/{n_nodes:,} nodes on cuda\n"
    else:
        line = (
            f"residency  {n_fast:,}/{n_nodes:,} nodes on cuda, "
            f"{n_nodes - n_fast:,} nodes in ram\n"
        )
    sys.stderr.write(line)
    sys.stderr.flush()


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


def _eval_rate_torch_once(I, I_grid, r_grid, I_onset):
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


def _activation_chunk(n: int, device) -> int:
    """How many currents to interpolate at once without asking for a full-graph temporary."""
    n = int(n)
    if n <= _ACTIVATION_CHUNK:
        return n
    torch = _require_torch()
    if getattr(device, "type", None) != "cuda":
        return _ACTIVATION_CHUNK
    free, _total = torch.cuda.mem_get_info(device)
    # index int64 + several float64 gathers, kept under a fifth of whatever is still free.
    room = max(int(free) // 5, 8 << 20)
    return max(1 << 14, min(n, room // 128))


def _eval_rate_torch(I, I_grid, r_grid, I_onset):
    torch = _require_torch()
    try:
        return _eval_rate_torch_once(I, I_grid, r_grid, I_onset)
    except torch.OutOfMemoryError:
        if int(I.numel()) <= 1024:
            raise
        if I.device.type == "cuda":
            torch.cuda.empty_cache()
        mid = int(I.shape[0]) // 2
        r0, o0 = _eval_rate_torch(I[:mid], I_grid, r_grid, I_onset)
        r1, o1 = _eval_rate_torch(I[mid:], I_grid, r_grid, I_onset)
        return torch.cat((r0, r1)), o0 + o1


def _write_rate(I_total, idx, r_new, I_grid, r_grid, I_onset, chunk: int | None = None):
    """Evaluate ``F`` at ``idx`` and scatter into ``r_new``, in index-sized pieces."""
    torch = _require_torch()
    n = int(idx.numel())
    if n == 0:
        return torch.zeros((), dtype=torch.int64, device=I_total.device)
    step = _activation_chunk(n, I_total.device) if chunk is None else max(1, min(int(chunk), n))
    if step >= n:
        rt, n_oor = _eval_rate_torch(I_total[idx], I_grid, r_grid, I_onset)
        r_new[idx] = rt
        return n_oor
    oor = torch.zeros((), dtype=torch.int64, device=I_total.device)
    for start in range(0, n, step):
        part = idx[start : start + step]
        rt, n_oor = _eval_rate_torch(I_total[part], I_grid, r_grid, I_onset)
        r_new[part] = rt
        oor = oor + n_oor
    return oor


def _upload_array(values: np.ndarray, torch_dtype, device, label: str, unit: str):
    torch = _require_torch()
    src = np.ascontiguousarray(values)
    progress = get_progress()
    if not progress.enabled or src.size <= progress.chunk:
        return torch.as_tensor(src, dtype=torch_dtype, device=device)
    out = torch.empty(src.shape, dtype=torch_dtype, device=device)
    flat_out = out.reshape(-1)
    flat_src = src.reshape(-1)
    for start, end in iter_progress(int(flat_src.size), label, unit, progress.chunk):
        flat_out[start:end] = torch.as_tensor(flat_src[start:end], dtype=torch_dtype, device=device)
    return out


def _csr_to_torch(graph: GraphData, device):
    torch = _require_torch()
    csr = get_csr(graph)
    n = int(graph.n_nodes)
    crow = _upload_array(np.asarray(csr.indptr, dtype=np.int64), torch.int64, device, "uploading nodes", "nodes")
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float32)
    progress = get_progress()
    if not progress.enabled or indices.size <= progress.chunk:
        col = torch.as_tensor(indices, device=device)
        val = torch.as_tensor(data, device=device)
    else:
        col = torch.empty(indices.shape, dtype=torch.int64, device=device)
        val = torch.empty(data.shape, dtype=torch.float32, device=device)
        for start, end in iter_progress(int(indices.size), "uploading edges", "edges", progress.chunk):
            col[start:end] = torch.as_tensor(indices[start:end], device=device)
            val[start:end] = torch.as_tensor(data[start:end], device=device)
    return torch.sparse_csr_tensor(crow, col, val, size=(n, n), device=device, dtype=torch.float32)


def _table_tensors(table: FittedActivation, device) -> tuple[Any, Any, Any]:
    torch = _require_torch()
    return (
        torch.as_tensor(np.ascontiguousarray(table.I_grid, dtype=np.float64), device=device),
        torch.as_tensor(np.ascontiguousarray(table.r_grid, dtype=np.float64), device=device),
        torch.tensor(float(table.I_onset), dtype=torch.float64, device=device),
    )


def _prepare_split(
    graph: GraphData,
    groups: list[tuple[FittedActivation, np.ndarray]],
    dev_fast,
    n_fast: int,
    token: tuple[Any, ...],
) -> dict[str, Any]:
    torch = _require_torch()
    csr = get_csr(graph)
    n = int(graph.n_nodes)
    k = int(n_fast)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float32)
    nnz = int(indptr[k])
    dev_slow = torch.device("cpu")
    crow_f = _upload_array(indptr[: k + 1], torch.int64, dev_fast, "uploading nodes", "nodes")
    col_f = _upload_array(indices[:nnz], torch.int64, dev_fast, "uploading edges", "edges")
    val_f = _upload_array(data[:nnz], torch.float32, dev_fast, "uploading edges", "edges")
    csr_fast = torch.sparse_csr_tensor(
        crow_f, col_f, val_f, size=(k, n), device=dev_fast, dtype=torch.float32
    )
    crow_s = _upload_array(indptr[k:] - indptr[k], torch.int64, dev_slow, "staging nodes", "nodes")
    col_s = _upload_array(indices[nnz:], torch.int64, dev_slow, "staging edges", "edges")
    val_s = _upload_array(data[nnz:], torch.float32, dev_slow, "staging edges", "edges")
    csr_slow = torch.sparse_csr_tensor(
        crow_s, col_s, val_s, size=(n - k, n), device=dev_slow, dtype=torch.float32
    )
    idx_fast: list[Any] = []
    idx_slow: list[Any] = []
    grids_fast: list[tuple[Any, Any, Any]] = []
    grids_slow: list[tuple[Any, Any, Any]] = []
    for table, idx in groups:
        flat = np.asarray(idx, dtype=np.int64)
        on_fast = flat[flat < k]
        on_slow = flat[flat >= k] - k
        idx_fast.append(torch.as_tensor(on_fast, dtype=torch.int64, device=dev_fast))
        idx_slow.append(torch.as_tensor(on_slow, dtype=torch.int64, device=dev_slow))
        grids_fast.append(_table_tensors(table, dev_fast))
        grids_slow.append(_table_tensors(table, dev_slow))
    bundle: dict[str, Any] = {
        "token": token,
        "split": True,
        "device": dev_fast,
        "n_fast": k,
        "csr_fast": csr_fast,
        "csr_slow": csr_slow,
        "idx_fast": idx_fast,
        "idx_slow": idx_slow,
        "grids_fast": grids_fast,
        "grids_slow": grids_slow,
    }
    graph._torch = bundle
    return bundle


def prepare_torch_graph(
    graph: GraphData,
    tables: list[FittedActivation],
    device: str | None = None,
    vram_budget: int | None = None,
    *,
    _allow_spill: bool = True,
) -> dict[str, Any]:
    torch = _require_torch()
    dev = _device(torch, device)
    n = int(graph.n_nodes)
    n_fast = n
    if _allow_spill and dev.type == "cuda" and n > 0:
        ptr = np.asarray(get_csr(graph).indptr)
        budget = cuda_vram_budget(torch) if vram_budget is None else int(vram_budget)
        n_fast = plan_gpu_rows(ptr, n, budget)
        _report_residency(n_fast, n)
        if n_fast <= 0:
            return prepare_torch_graph(
                graph, tables, device="cpu", vram_budget=vram_budget, _allow_spill=False
            )
        if n_fast < n:
            prepare_graph(graph, n_types=len(tables))
            from nrde.engine.rate import coalesce_type_tables

            groups = coalesce_type_tables(tables, get_type_indices(graph, len(tables)))
            token = (id(graph.edge_index), id(graph.edge_weight), n, str(dev), n_fast, "f64-act")
            cached = graph._torch
            if isinstance(cached, dict) and cached.get("token") == token:
                return cached
            return _prepare_split(graph, groups, dev, n_fast, token)
    token = (id(graph.edge_index), id(graph.edge_weight), n, str(dev), n, "f64-act")
    cached = graph._torch
    if isinstance(cached, dict) and cached.get("token") == token:
        return cached
    prepare_graph(graph, n_types=len(tables))
    from nrde.engine.rate import coalesce_type_tables

    groups = coalesce_type_tables(tables, get_type_indices(graph, len(tables)))
    csr = _csr_to_torch(graph, dev)
    idx = []
    for _table, group_idx in groups:
        idx.append(
            _upload_array(np.asarray(group_idx, dtype=np.int64), torch.int64, dev, "uploading nodes", "nodes")
        )
    bundle: dict[str, Any] = {
        "token": token,
        "split": False,
        "device": dev,
        "csr": csr,
        "idx": idx,
        "I_grid": [],
        "r_grid": [],
        "I_onset": [],
    }
    for table, _idx in groups:
        # float64: float32 I_onset rounding breaks ``I < I_onset`` vs numpy at grid knots.
        grids = _table_tensors(table, dev)
        bundle["I_grid"].append(grids[0])
        bundle["r_grid"].append(grids[1])
        bundle["I_onset"].append(grids[2])
    graph._torch = bundle
    return bundle


def step_rate_torch(
    r: Any,
    I_ext: Any,
    bundle: dict[str, Any],
    oor: Any,
    dt: float = 1.0,
) -> tuple[Any, Any]:
    torch = _require_torch()
    csr = bundle["csr"]
    # SpMV in float32; promote before F so onset/interp match numpy float64.
    # r is Hz; W is nA/spike. Expected spikes in this step = r * dt_ms / 1000.
    spikes = r * (float(dt) / 1000.0)
    I_syn = torch.sparse.mm(csr, spikes.to(dtype=torch.float32).unsqueeze(1)).squeeze(1).to(dtype=torch.float64)
    I_total = I_syn + I_ext.to(dtype=torch.float64)
    r_new = torch.zeros(r.shape, dtype=torch.float64, device=r.device)
    for idx, I_grid, r_grid, I_onset in zip(
        bundle["idx"], bundle["I_grid"], bundle["r_grid"], bundle["I_onset"], strict=True
    ):
        if idx.numel() == 0:
            continue
        n_oor = _write_rate(I_total, idx, r_new, I_grid, r_grid, I_onset)
        oor = oor + n_oor
    return r_new, oor


def _as_oor(value: Any) -> int:
    torch = _require_torch()
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _step_split(r_fast, r_slow, I_fast, I_slow, bundle: dict[str, Any], oor: Any, dt: float = 1.0):
    """One step with the leading rows on ``r_fast``'s device and the rest in RAM."""
    torch = _require_torch()
    k = int(bundle["n_fast"])
    scale = float(dt) / 1000.0
    I_syn_f = torch.sparse.mm(
        bundle["csr_fast"], (r_fast * scale).to(dtype=torch.float32).unsqueeze(1)
    ).squeeze(1).to(dtype=torch.float64)
    I_syn_s = torch.sparse.mm(
        bundle["csr_slow"], (r_slow * scale).to(dtype=torch.float32).unsqueeze(1)
    ).squeeze(1).to(dtype=torch.float64)
    I_f = I_syn_f + I_fast
    I_s = I_syn_s + I_slow
    r_new_f = torch.zeros(k, dtype=torch.float64, device=r_fast.device)
    r_new_s = torch.zeros(int(r_slow.shape[0]) - k, dtype=torch.float64, device=r_slow.device)
    oor_i = _as_oor(oor)
    groups = zip(
        bundle["idx_fast"],
        bundle["idx_slow"],
        bundle["grids_fast"],
        bundle["grids_slow"],
        strict=True,
    )
    for idx_f, idx_s, grids_f, grids_s in groups:
        if idx_f.numel():
            oor_i += _as_oor(_write_rate(I_f, idx_f, r_new_f, *grids_f))
        if idx_s.numel():
            oor_i += _as_oor(_write_rate(I_s, idx_s, r_new_s, *grids_s))
    synced_fast = torch.empty(r_fast.shape, dtype=torch.float64, device=r_fast.device)
    synced_fast[:k] = r_new_f
    synced_fast[k:] = r_new_s.to(device=r_fast.device)
    synced_slow = torch.empty(r_slow.shape, dtype=torch.float64, device=r_slow.device)
    synced_slow[:k] = r_new_f.to(device=r_slow.device)
    synced_slow[k:] = r_new_s
    return synced_fast, synced_slow, oor_i


def run_rate_torch(
    graph: GraphData,
    tables: list[FittedActivation],
    n_steps: int,
    I_ext: np.ndarray | None = None,
    dt: float = 1.0,
    r0: float = 0.0,
    record_trace: bool = True,
    device: str | None = None,
    vram_budget: int | None = None,
) -> tuple[RateState, np.ndarray]:
    torch = _require_torch()
    bundle = prepare_torch_graph(graph, tables, device=device, vram_budget=vram_budget)
    I_np = np.zeros(graph.n_nodes, dtype=np.float64) if I_ext is None else np.asarray(I_ext, dtype=np.float64)
    trace_np = np.zeros((n_steps if record_trace else 0, graph.n_nodes), dtype=np.float64)
    progress = get_progress()
    if bundle.get("split"):
        k = int(bundle["n_fast"])
        dev = bundle["device"]
        r_fast = torch.full((graph.n_nodes,), float(r0), dtype=torch.float64, device=dev)
        r_slow = torch.full((graph.n_nodes,), float(r0), dtype=torch.float64, device="cpu")
        I_fast = torch.as_tensor(I_np[:k], dtype=torch.float64, device=dev)
        I_slow = torch.as_tensor(I_np[k:], dtype=torch.float64, device="cpu")
        oor: Any = 0
        for t in range(n_steps):
            r_fast, r_slow, oor = _step_split(r_fast, r_slow, I_fast, I_slow, bundle, oor, dt=dt)
            if record_trace:
                trace_np[t, :k] = r_fast[:k].detach().cpu().numpy()
                trace_np[t, k:] = r_slow[k:].detach().cpu().numpy()
            progress.tick(t + 1, n_steps, "rate steps", "steps")
        r_host = np.empty(graph.n_nodes, dtype=np.float64)
        r_host[:k] = r_fast[:k].detach().cpu().numpy()
        r_host[k:] = r_slow[k:].detach().cpu().numpy()
        return RateState(r=r_host, out_of_range=_as_oor(oor)), trace_np
    dev = bundle["device"]
    I_t = torch.as_tensor(np.ascontiguousarray(I_np), device=dev)
    r = torch.full((graph.n_nodes,), float(r0), dtype=torch.float64, device=dev)
    oor = torch.zeros((), dtype=torch.int64, device=dev)
    for t in range(n_steps):
        r, oor = step_rate_torch(r, I_t, bundle, oor, dt=dt)
        if record_trace:
            trace_np[t] = r.detach().cpu().numpy()
        progress.tick(t + 1, n_steps, "rate steps", "steps")
    r_host = r.detach().cpu().numpy().astype(np.float64, copy=False)
    return RateState(r=r_host, out_of_range=_as_oor(oor)), trace_np
