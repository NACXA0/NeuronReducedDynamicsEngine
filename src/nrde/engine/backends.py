"""Online engine backend selector: numpy CSR (reference) vs torch cuSPARSE."""

from __future__ import annotations

import os


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_engine_backend(backend: str | None = None) -> str:
    """Resolve ``numpy`` or ``torch``. Default is numpy unless ``NRDE_ENGINE`` is set."""
    raw = backend if backend is not None else os.environ.get("NRDE_ENGINE", "numpy")
    name = str(raw or "numpy").strip().lower()
    if name in {"", "auto"}:
        if torch_cuda_available():
            return "torch"
        return "numpy"
    if name in {"numpy", "scipy", "cpu"}:
        return "numpy"
    if name in {"torch", "pytorch", "cuda"}:
        return "torch"
    raise ValueError(f"Unknown engine backend {name!r}. Use 'numpy' or 'torch'.")
