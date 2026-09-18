# 拟合产物（数据平面）

| 约定 | 说明 |
|---|---|
| **类型键（现行）** | `{type_id}.npz`，例如 `aCC.npz`、`type_0.npz`（`artifact_stem_for_type`） |
| 可选层级后缀 | `{type_id}_{layer}.npz`（`migrate --include-layer`） |
| 遗留模型键 | `adexp.npz` / `explif.npz` / `hh.npz` — `resolve_artifact_path` 仍可回退，但会 `DeprecationWarning` |
| 元数据 | `*.meta.json` 含 `type_ids` / `layer` / `git_commit` / `config_hash` / `naming=type_key` |

```bash
# 新拟合默认写类型键
nrde offline fit --model adexp --type-id aCC
# → artifacts/aCC.npz

# 一次性迁移目录
python scripts/migrate_artifacts_to_type_keys.py --dry-run
python scripts/migrate_artifacts_to_type_keys.py --keep-legacy
```
