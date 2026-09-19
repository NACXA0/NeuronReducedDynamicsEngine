# 数据源清单（需求 §2.4 / RFC-001 D12）

| 管线 | 数据源 | 服务模块 | 对拟合 F 的作用 |
|---|---|---|---|
| 连接组侧 | MaleCNS Feather、FlyWire | DD-1 图结构、FR-1.5 权重、DD-6 α 标定 | 无（不产生 f-I） |
| 电生理侧 | ModelDB、NeuroElectro、文献单类记录 | DD-2 参数、DD-3 拟合原料 | 直接决定 F 质量 |

更多连接组 = 更多可标定回路，**不**增加 f-I 曲线数量。

## 本地路径

- 约定见 [`data/README.md`](../data/README.md)
- 下载脚本：`uv run python scripts/download_malecns.py`（官方 GCS：`flyem-male-cns/v1.0/...`，CC-BY）
- 三件套：`connectome-weights-*.feather`、`body-annotations-*.feather`、`body-neurotransmitters-*.feather`

## fetch 与 manifest（RFC-002 D20）

A2 用户用统一入口拉预设数据，而不手抄 URL：

```bash
uv run nrde fetch fly --tier smoke
uv run nrde fetch fly --tier seed
uv run nrde fetch fly --tier full
uv run nrde fetch fly --refresh
```

| 档 | 内容 | 体积量级 |
|---|---|---|
| `smoke` | 校验已跟踪的 `artifacts/*.npz` sha256 | ≈0 |
| `seed` | 类型键种子（`type_0` / `aCC` / `hh_demo`） | MB |
| `full` | 种子 + MaleCNS Feather（weights 可选） | GB |

清单：`configs/manifests/fly.yaml`。缓存目录默认 `data/` 与 `artifacts/`；校验失败非零退出。Zenodo DOI 在 v0.1 发布时填入 manifest（当前可为空，回退 GCS / 仓库内种子）。
