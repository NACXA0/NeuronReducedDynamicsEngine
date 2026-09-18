# 拟合产物（数据平面）

| 约定 | 说明 |
|---|---|
| M1（当前） | 文件名可以是**模型键**：`adexp.npz` / `explif.npz` / `hh.npz`（验证对象是模型本身） |
| M2 起 | 迁移为**类型键**：`{type_id}.npz`，meta 含 `layer` / `type_ids` / `git_commit` / `config_hash` |
| 入库策略 | 小体积 L0 种子产物可入库；全类型 / L2 核族走 Release 附件或下载脚本 |
| 元数据 | 每个 `*.npz` 旁有 `*.meta.json`（schema_version ≥ 2） |

二进制 `*.npz` 默认 gitignore；需要种子产物时用：

```bash
fre fit --model adexp --out artifacts/adexp.npz
```
