"""Fetch downloads a missing file from the manifest's secondary zip URL."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
import yaml

from nrde.fetch import fetch_preset


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(root: Path, *, primary: str, secondary: str, digest: str, tier: str = "seed") -> Path:
    manifest = {
        "preset": "fly",
        "version": "test",
        "downloads": {"primary": primary, "secondary": secondary},
        "tiers": {
            tier: {
                "artifacts": [{"path": "artifacts/tiny.npz", "sha256": digest}],
            }
        },
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_secondary_zip_fills_a_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"not-a-real-npz-but-hashed"
    digest = _sha256(payload)
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("artifacts/tiny.npz", payload)
    root = tmp_path / "work"
    root.mkdir()
    monkeypatch.setenv("NRDE_ROOT", str(root))
    manifest = _write_manifest(
        tmp_path,
        primary="file:///does/not/exist.zip",
        secondary=bundle.as_uri(),
        digest=digest,
    )
    report = fetch_preset("fly", tier="seed", manifest_path=manifest)
    assert report["ok"] is True
    assert report["summary"] == {"verified": 0, "downloaded": 1}
    art = report["artifacts"][0]
    assert art["action"] == "downloaded"
    assert art["source"] == "secondary"
    assert art["status"] == "ok"
    assert (root / "artifacts" / "tiny.npz").read_bytes() == payload


def test_local_file_is_verified_without_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"already-here"
    digest = _sha256(payload)
    root = tmp_path / "work"
    dest = root / "artifacts" / "tiny.npz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(payload)
    monkeypatch.setenv("NRDE_ROOT", str(root))
    manifest = _write_manifest(tmp_path, primary="", secondary="", digest=digest)
    report = fetch_preset("fly", tier="seed", manifest_path=manifest)
    assert report["ok"] is True
    assert report["summary"]["verified"] == 1
    assert report["summary"]["downloaded"] == 0
    assert report["artifacts"][0]["action"] == "verified"
    assert report["downloads"] == []


def test_missing_file_without_urls_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "work"
    root.mkdir()
    monkeypatch.setenv("NRDE_ROOT", str(root))
    manifest = _write_manifest(tmp_path, primary="", secondary="", digest="abc", tier="pretrained")
    report = fetch_preset("fly", tier="pretrained", manifest_path=manifest)
    assert report["ok"] is False
    assert report["artifacts"][0]["status"] == "missing_seed"
    assert "downloads.secondary" in report["artifacts"][0]["hint"]


def test_smoke_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"should-stay-in-the-zip"
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("artifacts/tiny.npz", payload)
    root = tmp_path / "work"
    root.mkdir()
    monkeypatch.setenv("NRDE_ROOT", str(root))
    manifest = _write_manifest(
        tmp_path,
        primary="",
        secondary=bundle.as_uri(),
        digest=_sha256(payload),
        tier="smoke",
    )
    report = fetch_preset("fly", tier="smoke", manifest_path=manifest)
    assert report["ok"] is False
    assert report["artifacts"][0]["status"] == "missing"
    assert not (root / "artifacts" / "tiny.npz").exists()
