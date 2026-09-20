# 数据集下载地址

用户不手改 URL。他们只运行 `nrde fetch`。地址写在 `configs/manifests/fly.yaml` 顶部：

```yaml
downloads:
  primary: ""
  secondary: ""
  note: "..."
```

| 字段 | 含义 |
|---|---|
| `primary` | 第一跳。约定放 GitHub Release 附件 `nrde-fly-offline.zip` 的直链。 |
| `secondary` | 第二跳。`primary` 为空、下载失败，或 zip 里没有该成员时才用。约定放 Zenodo 或任何其它托管上的**同一结构**的 zip。 |
| `note` | 给人看的。导出脚本重写 YAML 时不会丢掉这个键。 |

空字符串表示跳过这一跳。两个都空时，`fetch` 只校验工作区里已有的文件；缺文件则失败，且不会访问网络。`smoke` 档永远不下载。

zip 内路径必须与清单一致，例如 `datasets/fly/fits/aCC_l0.npz`、`artifacts/aCC.npz`。`scripts/make_release.py` 打出来的就是这种布局。

## 改第二地址

1. 把同一份 `dist/nrde-fly-offline.zip` 传到镜像（Zenodo 记录或其它 HTTPS 直链）。记录的是文件 URL，不是概念 DOI 页面；`fetch` 用的是 `urllib` 直接取 zip。
2. 只改 `secondary` 的字符串。不要改各文件的 sha256，除非表本身变了（那是 [pretrained.md](pretrained.md)）。
3. 提交 `configs/manifests/fly.yaml`。
4. 验证时不要依赖当前这份已经齐的工作区。另找一个目录，设置 `NRDE_ROOT` 指向空目录，并让清单里的 `primary` 暂时指向一个不存在的地址（或留空），`secondary` 指向新 URL：

```bash
mkdir -p /tmp/nrde-fetch-check
NRDE_ROOT=/tmp/nrde-fetch-check uv run nrde fetch fly --tier pretrained --manifest configs/manifests/fly.yaml
```

若你没有临时改 primary，而本仓库里文件都在，这条命令会显示「已校验，未下载」，**不能**证明第二地址可用。要证明第二地址，primary 必须失败或缺文件。

成功时标准错误是 `pretrained: 0 已校验，33 已下载`（或只缺的那几个被下载）。JSON 里对应条目的 `source` 为 `secondary`，`action` 为 `downloaded`。

## 改主地址

发新 Release 后，把 `primary` 换成新 tag 的附件 URL。`secondary` 可以继续指向上一版镜像，直到镜像也更新。两份 zip 的成员哈希必须和清单一致，否则 `fetch` 报 `hash_mismatch` 并返回非 0。

缓存放在 `data/cache/`（已随 `data/**` gitignore）。换 URL 会换缓存文件名。旧缓存可以删。
