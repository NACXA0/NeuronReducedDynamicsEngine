# 数据源清单（需求 §2.4 / RFC-001 D12）

| 管线 | 数据源 | 服务模块 | 对拟合 F 的作用 |
|---|---|---|---|
| 连接组侧 | MaleCNS Feather、FlyWire | DD-1 图结构、FR-1.5 权重、DD-6 α 标定 | 无（不产生 f-I） |
| 电生理侧 | ModelDB、NeuroElectro、文献单类记录 | DD-2 参数、DD-3 拟合原料 | 直接决定 F 质量 |

更多连接组 = 更多可标定回路，**不**增加 f-I 曲线数量。

## 教师槽

\(F_{\mathrm{type}}\) 的来源记在类型表的 `teacher` 里，不是拟合的一维。

| `kind` | 含义 |
|---|---|
| `builtin` | 本库点神经元 ODE（LIF / ExpLIF / AdExp / HH）。当前 literature 档（T4、aCC）仍是这个，notes 标明待 ModelDB。 |
| `modeldb` | 换教师后重拟合同一 `type_id`。写 accession，不把 ModelDB 的细胞 ID 写进 MaleCNS 的 `bodyId`。 |
| `fbl` | FlyBrainLab / 多房室轨迹，同样只升级对得上的类型。本轮没有 FBL 产物，也不跑容器。 |

脑区修正和位置修正没有数据，不写入发布 npz，已有类型表不必重训。要把它们挂进求值，见 [自定义管道](quickstart/extend.md)。换教师才会改表；种子 `artifacts/*.npz` 与 `datasets/fly/fits/` 的哈希不变。

## 本地路径

- 约定见 [`data/README.md`](../data/README.md)
- 下载脚本：`uv run python scripts/download_malecns.py`（官方 GCS：`flyem-male-cns/v1.0/...`，CC-BY）
- 三件套：`connectome-weights-*.feather`、`body-annotations-*.feather`、`body-neurotransmitters-*.feather`

## fetch 与 manifest（RFC-002 D20）

A2 用户用统一入口拉预设数据，而不手抄 URL：

```bash
uv run nrde fetch fly --tier smoke
uv run nrde fetch fly --tier seed
uv run nrde fetch fly --tier pretrained
uv run nrde fetch fly --tier prefer
uv run nrde fetch fly --tier full
uv run nrde fetch fly --refresh
```

| 档 | 内容 | 体积量级 |
|---|---|---|
| `smoke` | 校验已跟踪的 `artifacts/*.npz` sha256。不下载 | ≈0 |
| `seed` | 类型键种子（`type_0` / `aCC` / `hh_demo`） | MB |
| `pretrained` | 发布默认表 `datasets/fly/`（33 个 npz） | MB |
| `prefer` | 可选。效率 / 中性 / 拟真三份表，不进默认 pretrained | MB |
| `full` | 种子 + MaleCNS Feather（weights 可选，来自官方 GCS） | GB |

文件已在且哈希相符时，标准错误是「已校验，未下载」。缺文件时按清单里的地址下载 zip，先主地址，再第二地址；都空着则失败。清单是 `configs/manifests/fly.yaml`。校验失败非零退出。全量连接组的下载地址不在这份清单里，见 [`data/README.md`](../data/README.md)。
