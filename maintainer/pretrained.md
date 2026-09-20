# 更新预训练表

发布表在 `datasets/fly/`。种子在 `artifacts/type_0.npz`、`aCC.npz`、`hh_demo.npz`。两边不要混：示例拟合写到 `/tmp/nrde_fit/`，不要覆盖种子。`examples/01_single_neuron_fit.py` 会写 `artifacts/aCC.npz`，跑完若哈希变了，`fetch --tier seed` 会失败，用 git 恢复或按下面更新清单。

`outputs/` 是 PySR 等工作目录，已 gitignore，不要提交。

## 默认发布表

需要本地 MaleCNS 注释与递质（约 55 MB，不要权重也能抽附属回路）：

```bash
uv run python scripts/download_malecns.py --skip-weights
uv run python scripts/export_fly_default_dataset.py
```

脚本会重写：

- `datasets/fly/fits/`、`precision/`、`sparse/`、`reports/`
- `datasets/fly/manifest.json`（含每个文件的字节数和 sha256，以及 `full_sparse` 里三件 Feather 的登记）
- `configs/manifests/fly.yaml` 的 `tiers.pretrained`（只保留 `.npz`）

`downloads` 块会被原样保留（脚本是读入整份 YAML 再替换 `pretrained`）。若你手改 YAML 时把 `downloads` 删了，按 [downloads.md](downloads.md) 加回去。

然后：

```bash
uv run nrde fetch fly --tier pretrained
```

必须是「已校验，未下载」，且 `ok` 为 true。33 是当前预训练档的文件数，变了就要同时改 `tests/test_a1_a2.py`、`tests/test_evidence_ledger.py` 和证据链文档里的这个数字。

## 可选的效率 / 中性 / 拟真

不进默认 `pretrained` 档。

```bash
uv run python scripts/export_prefer_datasets.py
uv run nrde fetch fly --tier prefer
```

脚本只替换 `tiers.prefer`。

## 证据链

表或墙钟变了才重算。重算会改 `docs/reports/figures/` 的 CSV、SVG 和 `evidence.json`，墙钟随机器变。

```bash
uv run python scripts/make_figures.py
uv run python scripts/make_figures.py --verify
```

核对 README 对比表是否仍引用账本里的 \(R^2\)、RMSE 和墙钟。`tests/test_evidence_ledger.py` 会检查这一点。

## 种子

种子是小文件，继续进 Git，这样没填下载地址时克隆也能跑 demo。改种子后同步：

- `configs/manifests/fly.yaml` 里 `smoke`、`seed`、`full` 的 sha256
- `configs/flygym_demo.yaml` 里的 `artifact_hashes`（若 demo 在用）

不要为了体积把 `datasets/fly/**/*.npz` 加进 gitignore，除非 `downloads.primary` 已经指向一份含这些文件的 zip，并且你在空目录验证过 `fetch` 能下回来。
