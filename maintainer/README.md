# 维护者手册

这里只给本仓库的维护者。用户文档（README、`docs/`）不写发布步骤。

| 要做的事 | 页面 |
|---|---|
| 打 GitHub Release，并把离线 zip 挂上去 | [release.md](release.md) |
| 构建并上传 PyPI | [pypi.md](pypi.md) |
| 重做预训练表、种子 npz、证据链 | [pretrained.md](pretrained.md) |
| 修改数据集的主下载地址和第二下载地址 | [downloads.md](downloads.md) |
| 重新生成 GIF / Colab | [assets.md](assets.md) |
| 构建并挂出静态宣传页 | [WEB/README.md](WEB/README.md) |

现在还不要发版。`pyproject.toml` 与 Changelog 的已发布段仍是 0.1.0，未发布改动在 `[Unreleased]`。`downloads.primary` 和 `downloads.secondary` 都是空字符串：克隆用户靠 Git 里的表，`fetch` 只校验。地址填上之后，缺文件才会下载。

发版顺序：

1. 按 [pretrained.md](pretrained.md) 确认表和哈希。
2. 按 [release.md](release.md) 打 zip、写 Changelog、推 tag、把 zip 挂到 Release。
3. 把 Release 附件的 URL 写入 `downloads.primary`，镜像写入 `downloads.secondary`（[downloads.md](downloads.md)）。
4. 再按 [pypi.md](pypi.md) 上传。PyPI 的 wheel 不含 `datasets/`，装好之后要靠这两条 URL。
