"""Data preset fetch via manifest (RFC-002 D20): nrde fetch fly."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from nrde.paths import repo_root

_MANIFEST_NAME = Path("configs") / "manifests" / "fly.yaml"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    root = repo_root()
    p = Path(path) if path else root / _MANIFEST_NAME
    if not p.is_absolute():
        p = root / p
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _download_urls(raw: dict[str, Any]) -> list[tuple[str, str]]:
    block = raw.get("downloads") or {}
    found: list[tuple[str, str]] = []
    for key in ("primary", "secondary"):
        url = str(block.get(key) or "").strip()
        if url:
            found.append((key, url))
    return found


def _cache_zip(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    dest = cache_dir / f"{digest}.zip"
    if dest.is_file() and zipfile.is_zipfile(dest):
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
        if not zipfile.is_zipfile(tmp):
            raise ValueError(f"not a zip: {url}")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _extract_member(zf: zipfile.ZipFile, rel: str, root: Path) -> None:
    name = rel.replace("\\", "/")
    names = {info.filename.replace("\\", "/"): info for info in zf.infolist() if not info.is_dir()}
    info = names.get(name)
    if info is None:
        raise KeyError(name)
    dest = (root / name).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError(f"refusing to extract {name} outside {root}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(zf.read(info))


class _ZipSource:
    """Lazy download of the first URL that yields a zip. Later files reuse it."""

    def __init__(self, urls: list[tuple[str, str]], cache_dir: Path) -> None:
        self.urls = urls
        self.cache_dir = cache_dir
        self.attempts: list[dict[str, Any]] = []
        self._open: list[tuple[str, zipfile.ZipFile]] = []
        self._failed: set[str] = set()

    def close(self) -> None:
        for _label, zf in self._open:
            zf.close()
        self._open.clear()

    def extract(self, rel: str, root: Path) -> str | None:
        for label, zf in self._open:
            try:
                _extract_member(zf, rel, root)
            except KeyError:
                continue
            return label
        for label, url in self.urls:
            if label in self._failed or any(existing == label for existing, _zf in self._open):
                continue
            try:
                path = _cache_zip(url, self.cache_dir)
                zf = zipfile.ZipFile(path)
            except (OSError, urllib.error.URLError, ValueError, zipfile.BadZipFile) as exc:
                self._failed.add(label)
                self.attempts.append({"source": label, "url": url, "status": "error", "error": str(exc)})
                continue
            self._open.append((label, zf))
            self.attempts.append({"source": label, "url": url, "status": "ok"})
            try:
                _extract_member(zf, rel, root)
            except KeyError:
                continue
            return label
        return None


def fetch_preset(
    name: str = "fly",
    *,
    tier: str = "seed",
    refresh: bool = False,
    manifest_path: str | Path | None = None,
    skip_weights: bool = True,
) -> dict[str, Any]:
    """Fetch or verify a named data preset. Returns a JSON-serializable report.

    Files already on disk with a matching sha256 are verified and not downloaded.
    Missing files (except ``smoke``) are taken from ``downloads.primary``, then
    ``downloads.secondary``. Both empty means the checkout must already contain them.
    """
    del name  # only 'fly' for now; manifest embeds preset id
    root = repo_root()
    raw = load_manifest(manifest_path)
    tiers = raw.get("tiers") or {}
    if tier not in tiers:
        raise KeyError(f"Unknown tier {tier!r}. Known: {sorted(tiers)}")
    spec = tiers[tier]
    urls = _download_urls(raw)
    source = _ZipSource(urls, root / "data" / "cache")
    report: dict[str, Any] = {
        "preset": raw.get("preset", "fly"),
        "tier": tier,
        "version": raw.get("version"),
        "root": str(root),
        "artifacts": [],
        "downloads": [],
        "connectome": None,
        "summary": {"verified": 0, "downloaded": 0},
        "ok": True,
    }

    try:
        for item in spec.get("artifacts") or []:
            rel = str(item["path"])
            path = root / rel
            entry: dict[str, Any] = {"path": rel, "exists": path.exists()}
            expected = item.get("sha256")
            if path.exists():
                digest = _sha256(path)
                entry["sha256"] = digest
                if expected and digest != expected:
                    if not refresh or tier == "smoke" or not urls:
                        entry["status"] = "hash_mismatch"
                        entry["action"] = "hash_mismatch"
                        entry["expected"] = expected
                        report["ok"] = False
                        report["artifacts"].append(entry)
                        continue
                else:
                    entry["status"] = "ok_refresh_noop" if refresh else "ok"
                    entry["action"] = "verified"
                    entry["source"] = "local"
                    report["summary"]["verified"] += 1
                    report["artifacts"].append(entry)
                    continue

            if tier == "smoke" or not urls:
                entry["exists"] = path.exists()
                entry["status"] = "missing" if tier == "smoke" else "missing_seed"
                entry["action"] = "missing"
                report["ok"] = False
                if tier != "smoke":
                    entry["hint"] = (
                        "File is not in the working tree and downloads.primary / "
                        "downloads.secondary are empty. Clone the full repository, "
                        "or set a zip URL in configs/manifests/fly.yaml."
                    )
                report["artifacts"].append(entry)
                continue

            label = source.extract(rel, root)
            entry["exists"] = path.exists()
            if label is None or not path.exists():
                entry["status"] = "missing_seed"
                entry["action"] = "missing"
                report["ok"] = False
                entry["hint"] = f"Could not extract {rel} from downloads.primary or downloads.secondary"
                report["artifacts"].append(entry)
                continue
            digest = _sha256(path)
            entry["sha256"] = digest
            entry["source"] = label
            if expected and digest != expected:
                entry["status"] = "hash_mismatch"
                entry["action"] = "hash_mismatch"
                entry["expected"] = expected
                report["ok"] = False
            else:
                entry["status"] = "ok"
                entry["action"] = "downloaded"
                report["summary"]["downloaded"] += 1
            report["artifacts"].append(entry)
    finally:
        report["downloads"] = source.attempts
        source.close()

    conn = spec.get("connectome")
    if conn:
        out_dir = root / str(conn.get("out_dir", "data/malecns"))
        script = root / str(conn.get("script", "scripts/download_malecns.py"))
        args = list(conn.get("args") or [])
        if skip_weights and "--skip-weights" not in args:
            args.append("--skip-weights")
        if refresh:
            args.append("--force")
        if not script.is_file():
            report["connectome"] = {
                "error": f"missing {script}",
                "hint": "tier=full runs scripts/download_malecns.py and needs a git checkout",
            }
            report["ok"] = False
        else:
            cmd = [sys.executable, str(script), "--out", str(out_dir), *args]
            try:
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                report["connectome"] = {
                    "cmd": cmd,
                    "returncode": proc.returncode,
                    "stdout_tail": (proc.stdout or "")[-500:],
                    "stderr_tail": (proc.stderr or "")[-500:],
                }
                if proc.returncode != 0:
                    report["ok"] = False
            except OSError as exc:
                report["connectome"] = {"error": str(exc)}
                report["ok"] = False

    report_path = root / "data" / "fetch_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
