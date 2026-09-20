# 发布 GitHub Release

离线包是 `dist/nrde-fly-offline.zip`。Git 不跟踪 `dist/`。包里是种子 `artifacts/*.npz`、`configs/flygym_demo.yaml`、`configs/manifests/fly.yaml`，以及当时工作区里整个 `datasets/fly/`。全量 MaleCNS Feather 不要打进去，那份约 1 GB 的权重继续走官方 GCS。

## 什么时候发

三件事同时成立再发，不要只为了「有一个 Release」：

- `[Unreleased]` 已移进新版本号，`pyproject.toml` 的 `version` 与 `src/nrde/__init__.py` 的 `__version__` 一致。
- `uv run nrde fetch fly --tier pretrained` 以「已校验，未下载」结束，33 个表哈希通过。
- 你准备在发出 Release 之后，立刻把附件 URL 写进清单（见 [downloads.md](downloads.md)）。URL 空着时，pip 安装的用户缺表就会失败。

0.1.0 已经写在 Changelog 里。下一版用 0.1.1，除非这次改动你明确算作 0.2.0。

## 构建 zip

在干净的检出上（预训练表都在）：

```bash
uv run python scripts/make_release.py --out dist/nrde-fly-offline.zip
```

脚本在缺种子或缺 `datasets/fly` 下的文件时返回非 0。zip 根上有 `manifest.json`（每个文件的 sha256）。

`.github/workflows/ci-demo-assets.yml` 会在 Release 事件上再打一次 zip，但只上传成 Actions artifact，不会自动挂到 Release 附件。附件要自己挂。

## 版本与说明

1. 编辑 `pyproject.toml` 的 `version`，以及 `src/nrde/__init__.py` 的 `__version__`。
2. 把 `docs/CHANGELOG.md` 的 `[Unreleased]` 改成 `## [x.y.z] — YYYY-MM-DD`，上面留一个新的空 `[Unreleased]`。
3. Release 正文按用户路径写，不要贴验收表。建议四行：
   - 克隆后：`uv sync`，然后 `uv run nrde fetch fly --tier pretrained`（应显示已校验、未下载），再 `uv run nrde run simulate --fit datasets/fly/fits/aCC_l0.npz --n 80 --steps 50`。
   - pip 安装后：在空目录运行同一条 `fetch`。表从本 Release 的 zip 下载（地址写进清单之后）。
   - 不装环境：Colab notebook 与 `assets/demo/` 的 GIF。
   - 自己拟合：`docs/quickstart/pipeline.md`。不要写回 `artifacts/aCC.npz`。

## 推 tag 并挂附件

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin HEAD
git push origin vX.Y.Z
gh release create vX.Y.Z dist/nrde-fly-offline.zip --title "vX.Y.Z" --notes-file /tmp/nrde-release-notes.md
```

附件 URL 形如：

```text
https://github.com/NACXA0/NeuronReducedDynamicsEngine/releases/download/vX.Y.Z/nrde-fly-offline.zip
```

把这个 URL 填进 `downloads.primary`。第二地址见 [downloads.md](downloads.md)。

发完后在另一份空目录验证：只留清单、删掉一个 `datasets/fly/fits/*.npz`，再 `uv run nrde fetch fly --tier pretrained`，确认那一个文件来自 `source: primary`，并且哈希通过。不要在唯一的工作副本上做这步。
