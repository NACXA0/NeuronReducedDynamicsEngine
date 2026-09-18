"""Migrate / resolve type-keyed artifacts (RFC-002 P2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nrde.fitting.artifacts import (
    default_artifact_path,
    is_model_keyed_stem,
    migrate_artifact,
    migrate_directory,
    resolve_artifact_path,
)
from nrde.fitting.fit import fit_fi, save_fit


def test_default_artifact_path_is_type_keyed():
    fit = fit_fi(
        "lif",
        type_id="aCC",
        I_min=0.0,
        I_max=0.5,
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
    )
    path = default_artifact_path(fit)
    assert path.name == "aCC.npz"
    assert not is_model_keyed_stem(path.stem)


def test_migrate_model_key_to_type_key(tmp_path: Path):
    fit = fit_fi(
        "explif",
        type_id="type_0",
        I_min=0.0,
        I_max=0.6,
        n_I=6,
        t_total=250.0,
        window=150.0,
        n_validate=2,
    )
    legacy = tmp_path / "explif.npz"
    save_fit(fit, legacy, skip_if_unchanged=False)
    report = migrate_artifact(legacy, keep_legacy=True)
    assert report["action"] == "copied"
    dst = tmp_path / "type_0.npz"
    assert dst.exists()
    assert legacy.exists()
    assert (tmp_path / "type_0.meta.json").exists()
    meta = (tmp_path / "type_0.meta.json").read_text(encoding="utf-8")
    assert '"naming": "type_key"' in meta


def test_migrate_idempotent_and_resolve(tmp_path: Path):
    fit = fit_fi(
        "adexp",
        type_id="aCC",
        I_min=0.0,
        I_max=0.7,
        n_I=6,
        t_total=250.0,
        window=150.0,
        n_validate=2,
    )
    legacy = tmp_path / "adexp.npz"
    save_fit(fit, legacy, skip_if_unchanged=False)
    migrate_artifact(legacy, keep_legacy=True)
    report2 = migrate_artifact(legacy, keep_legacy=True)
    assert report2["action"] in {"exists_same_hash", "copied"}

    found = resolve_artifact_path(tmp_path, type_id="aCC", model="adexp")
    assert found is not None
    assert found.name == "aCC.npz"

    # Type key missing → legacy model key with deprecation.
    only_legacy = tmp_path / "legacy_only"
    only_legacy.mkdir()
    save_fit(fit, only_legacy / "adexp.npz", skip_if_unchanged=False)
    with pytest.warns(DeprecationWarning, match="legacy model-keyed"):
        hit = resolve_artifact_path(only_legacy, type_id="aCC", model="adexp")
    assert hit is not None and hit.name == "adexp.npz"


def test_migrate_directory_skips_type_keyed(tmp_path: Path):
    fit = fit_fi(
        "lif",
        type_id="Kenyon",
        I_min=0.0,
        I_max=0.5,
        n_I=6,
        t_total=200.0,
        window=120.0,
        n_validate=2,
    )
    save_fit(fit, tmp_path / "Kenyon.npz", skip_if_unchanged=False)
    reports = migrate_directory(tmp_path, dry_run=True)
    assert reports[0]["action"] == "noop"
