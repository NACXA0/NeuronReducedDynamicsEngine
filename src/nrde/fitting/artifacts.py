"""Artifact path helpers: type-keyed names + legacy model-key fallback (RFC-002 P2)."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any

from nrde.fitting.fit import artifact_stem_for_type, load_fit
from nrde.types import FittedActivation

# Historical M1 filenames keyed by ODE family rather than cell type.
MODEL_KEY_STEMS: frozenset[str] = frozenset(
    {"lif", "explif", "adexp", "hh", "izhikevich", "fit", "toy_leak"}
)


def meta_path_for(npz_path: Path) -> Path:
    npz_path = Path(npz_path)
    if npz_path.suffix == ".npz":
        return Path(str(npz_path)[:-4] + ".meta.json")
    return npz_path.with_suffix(".meta.json")


def is_model_keyed_stem(stem: str) -> bool:
    return stem.lower() in MODEL_KEY_STEMS


def read_meta(npz_path: Path) -> dict[str, Any]:
    meta = meta_path_for(npz_path)
    if not meta.exists():
        return {}
    return json.loads(meta.read_text(encoding="utf-8"))


def type_id_from_artifact(npz_path: Path) -> str:
    """Prefer meta.type_id; fall back to stem if already type-keyed."""
    meta = read_meta(npz_path)
    tid = str(meta.get("type_id") or "").strip()
    if tid and tid not in {"unknown", "default", "demo"}:
        return tid
    ids = meta.get("type_ids") or []
    if ids:
        return str(ids[0])
    stem = Path(npz_path).stem
    if not is_model_keyed_stem(stem):
        return stem
    model = str(meta.get("model") or stem)
    return f"{model}_legacy"


def target_stem_for(
    npz_path: Path,
    *,
    include_layer: bool = False,
    type_id: str | None = None,
) -> str:
    meta = read_meta(npz_path)
    tid = type_id or type_id_from_artifact(npz_path)
    layer = str(meta.get("layer") or "") if include_layer else None
    if layer in {"", "None"}:
        layer = None
    return artifact_stem_for_type(tid, layer)


def default_artifact_path(
    fit: FittedActivation,
    root: str | Path = "artifacts",
    *,
    include_layer: bool = False,
) -> Path:
    """Canonical M2+ path for a fitted activation."""
    stem = artifact_stem_for_type(fit.type_id, fit.layer if include_layer else None)
    return Path(root) / f"{stem}.npz"


def resolve_artifact_path(
    root: str | Path,
    *,
    type_id: str | None = None,
    model: str | None = None,
    layer: str | None = None,
) -> Path | None:
    """Locate an artifact: type key (+ optional layer) first, then legacy model key."""
    root = Path(root)
    candidates: list[Path] = []
    if type_id:
        candidates.append(root / f"{artifact_stem_for_type(type_id)}.npz")
        if layer:
            candidates.append(root / f"{artifact_stem_for_type(type_id, layer)}.npz")
    if model:
        candidates.append(root / f"{model}.npz")
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            if is_model_keyed_stem(path.stem) and type_id and path.stem.lower() != type_id.lower():
                warnings.warn(
                    f"Using legacy model-keyed artifact {path.name}; "
                    f"prefer type-keyed {artifact_stem_for_type(type_id)}.npz "
                    "(run scripts/migrate_artifacts_to_type_keys.py)",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return path
    return None


def migrate_artifact(
    npz_path: str | Path,
    *,
    include_layer: bool = False,
    dry_run: bool = False,
    keep_legacy: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """
    Rename/copy a model-keyed npz (+ meta) to type-keyed names.

    Returns a report dict. Idempotent if already type-keyed.
    """
    src = Path(npz_path)
    if not src.exists():
        raise FileNotFoundError(src)
    meta_src = meta_path_for(src)
    stem = target_stem_for(src, include_layer=include_layer)
    dst = src.with_name(f"{stem}.npz")
    meta_dst = meta_path_for(dst)

    report: dict[str, Any] = {
        "src": str(src),
        "dst": str(dst),
        "type_id": type_id_from_artifact(src),
        "already_type_keyed": not is_model_keyed_stem(src.stem) and src.resolve() == dst.resolve(),
        "action": "skip",
    }

    if src.resolve() == dst.resolve():
        report["action"] = "noop"
        return report

    if dst.exists() and not force:
        # Same config_hash → treat as done; else refuse without --force.
        try:
            old = load_fit(src)
            new = load_fit(dst)
            if old.config_hash and old.config_hash == new.config_hash:
                report["action"] = "exists_same_hash"
                if not keep_legacy and not dry_run and is_model_keyed_stem(src.stem):
                    src.unlink(missing_ok=True)
                    meta_src.unlink(missing_ok=True)
                    report["removed_legacy"] = True
                return report
        except Exception:  # noqa: BLE001
            pass
        report["action"] = "exists_conflict"
        return report

    if dry_run:
        report["action"] = "would_migrate"
        return report

    if keep_legacy:
        shutil.copy2(src, dst)
        if meta_src.exists():
            shutil.copy2(meta_src, meta_dst)
        report["action"] = "copied"
    else:
        src.replace(dst)
        if meta_src.exists():
            meta_src.replace(meta_dst)
        report["action"] = "renamed"

    # Refresh meta naming stamp without rewriting arrays.
    if meta_dst.exists():
        meta = json.loads(meta_dst.read_text(encoding="utf-8"))
        meta["naming"] = "type_key"
        meta["legacy_stem"] = src.stem if is_model_keyed_stem(src.stem) else meta.get("legacy_stem")
        meta_dst.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return report


def migrate_directory(
    root: str | Path = "artifacts",
    *,
    include_layer: bool = False,
    dry_run: bool = False,
    keep_legacy: bool = True,
    force: bool = False,
) -> list[dict[str, Any]]:
    root = Path(root)
    reports: list[dict[str, Any]] = []
    for npz in sorted(root.glob("*.npz")):
        if not is_model_keyed_stem(npz.stem):
            # Still stamp naming if meta lacks it.
            reports.append(
                {
                    "src": str(npz),
                    "dst": str(npz),
                    "type_id": type_id_from_artifact(npz),
                    "already_type_keyed": True,
                    "action": "noop",
                }
            )
            continue
        reports.append(
            migrate_artifact(
                npz,
                include_layer=include_layer,
                dry_run=dry_run,
                keep_legacy=keep_legacy,
                force=force,
            )
        )
    return reports


def load_fit_resolved(
    root: str | Path,
    *,
    type_id: str | None = None,
    model: str | None = None,
    layer: str | None = None,
) -> FittedActivation:
    path = resolve_artifact_path(root, type_id=type_id, model=model, layer=layer)
    if path is None:
        raise FileNotFoundError(
            f"No artifact in {root!s} for type_id={type_id!r} model={model!r}"
        )
    return load_fit(path)


# Re-export intentionally omitted — callers import save_fit from nrde.fitting.fit.
__all__ = [
    "MODEL_KEY_STEMS",
    "default_artifact_path",
    "is_model_keyed_stem",
    "load_fit_resolved",
    "meta_path_for",
    "migrate_artifact",
    "migrate_directory",
    "read_meta",
    "resolve_artifact_path",
    "target_stem_for",
    "type_id_from_artifact",
]
