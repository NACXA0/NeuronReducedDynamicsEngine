"""Shared fixtures: synthetic connectome + standard current protocols."""

from __future__ import annotations

import numpy as np
import pytest

from nrde.fitting.fit import fit_fi
from nrde.io.connectome import erdos_renyi_graph


@pytest.fixture
def standard_I_grid() -> np.ndarray:
    """Common f-I current axis (nA) for unit tests."""
    return np.linspace(0.0, 0.8, 9)


@pytest.fixture
def standard_I_ext(n: int = 32) -> np.ndarray:
    """Constant external drive for small circuit steps."""
    return np.full(n, 0.35, dtype=np.float64)


@pytest.fixture
def synth_connectome():
    """Small ER graph with two type labels (no Feather download)."""
    return erdos_renyi_graph(
        32,
        p=0.08,
        n_types=2,
        seed=11,
        weight=0.001,
        type_names=("type_0", "type_1"),
    )


@pytest.fixture(scope="module")
def lif_fit_table():
    """Cheap LIF L0 table reused by engine / adapter tests."""
    return fit_fi(
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
